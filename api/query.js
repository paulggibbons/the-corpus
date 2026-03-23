import { kv } from "@vercel/kv";
import { createClient } from "@supabase/supabase-js";
import Anthropic from "@anthropic-ai/sdk";

const MAX_QUERY_LENGTH = 500;
const PER_IP_HOURLY = 20;
const PER_IP_DAILY = 50;
const GLOBAL_HOURLY = 500;
const ALERT_EMAIL = "paul@paulgibbonsadvisory.com";

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function getEmbedding(text) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: "text-embedding-3-small", input: text }),
  });
  const data = await res.json();
  return data.data[0].embedding;
}

function hourKey(prefix) {
  return `${prefix}:${new Date().toISOString().slice(0, 13)}`;
}
function dayKey(prefix) {
  return `${prefix}:${new Date().toISOString().slice(0, 10)}`;
}

async function checkRateLimits(ip) {
  const globalCount = (await kv.get(hourKey("global"))) || 0;
  if (globalCount >= GLOBAL_HOURLY) {
    sendAlert(globalCount).catch(() => {});
    return { allowed: false, message: "Service temporarily paused.", status: 503 };
  }
  const ipHourCount = (await kv.get(hourKey(`ip:${ip}`))) || 0;
  if (ipHourCount >= PER_IP_HOURLY) {
    return { allowed: false, message: "Query limit reached. Try again tomorrow.", status: 429 };
  }
  const ipDayCount = (await kv.get(dayKey(`ip:${ip}`))) || 0;
  if (ipDayCount >= PER_IP_DAILY) {
    return { allowed: false, message: "Query limit reached. Try again tomorrow.", status: 429 };
  }
  return { allowed: true };
}

async function incrementCounters(ip) {
  const gk = hourKey("global");
  const ihk = hourKey(`ip:${ip}`);
  const idk = dayKey(`ip:${ip}`);
  await Promise.all([
    kv.incr(gk).then(() => kv.expire(gk, 3600)),
    kv.incr(ihk).then(() => kv.expire(ihk, 3600)),
    kv.incr(idk).then(() => kv.expire(idk, 86400)),
  ]);
}

async function sendAlert(count) {
  console.error(`[CIRCUIT BREAKER] Global hourly limit hit: ${count}. Alert: ${ALERT_EMAIL}`);
  if (process.env.RESEND_API_KEY) {
    await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "The Corpus <alerts@paulgibbonsadvisory.com>",
        to: ALERT_EMAIL,
        subject: "[The Corpus] Circuit breaker triggered",
        text: `Global hourly limit (${GLOBAL_HOURLY}) exceeded. Count: ${count}. Endpoint paused until hour rolls over.`,
      }),
    });
  }
}

const SYSTEM_PROMPT = `You are The Corpus — an AI assistant that answers questions using Paul Gibbons' intellectual body of work.

RULES:
1. Only answer based on the provided passages. If passages don't contain relevant info, say so.
2. Cite sources at book/chapter level for every claim.
3. Use Paul's voice — direct, evidence-based, occasionally contrarian. Not corporate.
4. If asked about topics outside the corpus, redirect: "That's outside my training. I can speak to [related topic]."
5. Keep answers substantive but concise. No filler.
6. When passages show evolution of thinking, note it.`;

export default async function handler(req, res) {
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { question } = req.body || {};
    if (!question || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Question is required." });
    }
    if (question.length > MAX_QUERY_LENGTH) {
      return res.status(400).json({ error: `Query must be under ${MAX_QUERY_LENGTH} characters.` });
    }

    const ip = req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.headers["x-real-ip"] || "unknown";
    const rateCheck = await checkRateLimits(ip);
    if (!rateCheck.allowed) {
      return res.status(rateCheck.status).json({ error: rateCheck.message });
    }
    await incrementCounters(ip);

    const embedding = await getEmbedding(question.trim());
    const { data: chunks, error: searchError } = await supabase.rpc("match_chunks", {
      query_embedding: embedding,
      match_threshold: 0.72,
      match_count: 8,
    });

    if (searchError) {
      console.error("Supabase search error:", searchError);
      return res.status(500).json({ error: "Search failed." });
    }
    if (!chunks || chunks.length === 0) {
      return res.json({
        answer: "I couldn't find relevant passages for that question. Try rephrasing, or ask about one of the 15 themes.",
        sources: [],
      });
    }

    const context = chunks
      .map((c, i) => `[Passage ${i + 1}] (${c.book_title || "Unknown"}, ${c.chapter || ""})\n${c.content}`)
      .join("\n\n---\n\n");

    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-5-20250514",
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [{
        role: "user",
        content: `Here are the most relevant passages from the corpus:\n\n${context}\n\n---\n\nQuestion: ${question.trim()}\n\nAnswer based only on these passages. Cite sources.`,
      }],
    });

    const answer = message.content[0]?.text || "No response generated.";
    const sources = [...new Set(chunks.map(c => c.book_title).filter(Boolean))];
    return res.json({ answer, sources });
  } catch (err) {
    console.error("Query handler error:", err);
    return res.status(500).json({ error: "Something went wrong." });
  }
}
