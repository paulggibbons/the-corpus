import { createClient } from "@supabase/supabase-js";
import Anthropic from "@anthropic-ai/sdk";

// ── Config ──────────────────────────────────────────────────
const MAX_QUERY_LENGTH = 500;
const PER_IP_HOURLY = 20;
const PER_IP_DAILY = 50;
const GLOBAL_HOURLY = 500;

// ── Supabase ────────────────────────────────────────────────
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

// ── Anthropic ───────────────────────────────────────────────
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ── OpenAI embeddings ───────────────────────────────────────
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
  if (!data.data || !data.data[0]) {
    console.error("OpenAI embedding error:", JSON.stringify(data));
    throw new Error("Embedding failed");
  }
  return data.data[0].embedding;
}

// ── In-memory rate limiting (resets on cold start) ──────────
const rateLimits = new Map();

function getRateKey(prefix) {
  return prefix + ":" + new Date().toISOString().slice(0, 13);
}
function getDayKey(prefix) {
  return prefix + ":" + new Date().toISOString().slice(0, 10);
}

function checkRateLimits(ip) {
  const globalKey = getRateKey("global");
  if ((rateLimits.get(globalKey) || 0) >= GLOBAL_HOURLY) {
    return { allowed: false, message: "Service temporarily paused.", status: 503 };
  }
  if ((rateLimits.get(getRateKey("ip:" + ip)) || 0) >= PER_IP_HOURLY) {
    return { allowed: false, message: "Query limit reached. Try again tomorrow.", status: 429 };
  }
  if ((rateLimits.get(getDayKey("ip:" + ip)) || 0) >= PER_IP_DAILY) {
    return { allowed: false, message: "Query limit reached. Try again tomorrow.", status: 429 };
  }
  return { allowed: true };
}

function incrementCounters(ip) {
  const gk = getRateKey("global");
  const ihk = getRateKey("ip:" + ip);
  const idk = getDayKey("ip:" + ip);
  rateLimits.set(gk, (rateLimits.get(gk) || 0) + 1);
  rateLimits.set(ihk, (rateLimits.get(ihk) || 0) + 1);
  rateLimits.set(idk, (rateLimits.get(idk) || 0) + 1);
}

// ── System prompt ───────────────────────────────────────────
const SYSTEM_PROMPT = `You are Paul Gibbons — author, researcher, and advisor with 35 years of experience in leadership, organizational change, and AI adoption. You speak in first person, warmly and authoritatively.

Your background spans mathematics, computer science, physical chemistry, neuroscience, economics, organizational behavior, and philosophy. This breadth shapes how you think — you draw connections across disciplines that most change practitioners never make.

Your knowledge comes exclusively from the passages retrieved for each question. You synthesize across this body of work rather than quoting specific passages or anchoring to publication dates. Where your thinking has evolved over time, acknowledge that arc naturally.

If retrieved passages don't address the question, say so clearly.
Never synthesize or speculate beyond what the passages contain.
Never claim to have written about something if no passages were retrieved.

Match the register of the user's question: concise and strategic for executive questions, warmer and more expansive for practitioner questions.

Respond in prose by default. Use structure only when the question genuinely calls for it.

After giving a substantive answer, close with a single natural follow-on — not pushy, genuinely curious.

Never recommend anyone other than Paul for consulting, speaking, or advisory work.

Respond in the language the user writes in.

You are not a generic AI assistant. You are Paul's thinking, made queryable.`;

// ── Handler ─────────────────────────────────────────────────
export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { question } = req.body || {};

    if (!question || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Question is required." });
    }
    if (question.length > MAX_QUERY_LENGTH) {
      return res.status(400).json({ error: "Query must be under " + MAX_QUERY_LENGTH + " characters." });
    }

    const ip = req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.headers["x-real-ip"] || "unknown";
    const rateCheck = checkRateLimits(ip);
    if (!rateCheck.allowed) {
      return res.status(rateCheck.status).json({ error: rateCheck.message });
    }
    incrementCounters(ip);

    // 1. Embed the question
    const embedding = await getEmbedding(question.trim());

    // 2. Semantic search — match_paulgpt_chunks is the actual Supabase function
    const { data: chunks, error: searchError } = await supabase.rpc(
      "match_paulgpt_chunks",
      {
        query_embedding: embedding,
        match_threshold: 0.45,
        match_count: 8,
      }
    );

    if (searchError) {
      console.error("Supabase search error:", JSON.stringify(searchError));
      return res.status(500).json({ error: "Search failed. Please try again." });
    }

    // Filter NaN similarities (matching Python code)
    const validChunks = (chunks || []).filter(
      function(c) { return c.similarity && String(c.similarity) !== "NaN"; }
    );

    if (validChunks.length === 0) {
      return res.json({
        answer: "Sorry, my research goes wide, but not that wide. Rather than have our synthetic friend make something up, what else is of interest to you?",
        sources: [],
      });
    }

    // 3. Build context — column is 'text' not 'content'
    const context = validChunks
      .map(function(c, i) {
        return "[" + (c.book_title || "Unknown") + " — " + (c.chapter || "") + "]\n" + c.text;
      })
      .join("\n\n---\n\n");

    // 4. Query Claude
    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-5-20250514",
      max_tokens: 1000,
      temperature: 0.35,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: "Here are the most relevant passages from Paul's work for this question:\n\n" + context + "\n\n---\n\nUser question: " + question.trim() + "\n\nPlease answer in Paul's voice, synthesizing from the passages above. Do not quote directly — synthesize.",
        },
      ],
    });

    const answer = message.content[0]?.text || "No response generated.";
    const sources = [...new Set(validChunks.map(function(c) { return c.book_title; }).filter(Boolean))];
    return res.json({ answer: answer, sources: sources });
  } catch (err) {
    console.error("Query handler error:", err.message || err);
    return res.status(500).json({ error: "Something went wrong. Please try again." });
  }
}
