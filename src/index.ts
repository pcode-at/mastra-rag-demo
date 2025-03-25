import { MDocument } from "@mastra/rag";
import { Mastra } from "@mastra/core";
import { PgVector } from "@mastra/pg";
import { embedMany } from "ai";

import { researchAgent } from "./agents/researchAgent";
import { openai } from "@ai-sdk/openai";

const pgVector = new PgVector(process.env.POSTGRES_CONNECTION_STRING!);
const mastra = new Mastra({
  agents: { researchAgent },
  vectors: { pgVector },
});

// Load the paper
const paperUrl = "https://arxiv.org/html/1706.03762";
const response = await fetch(paperUrl);
const paperText = await response.text();

// Create document and chunk it
const doc = MDocument.fromText(paperText);
const chunks = await doc.chunk({
  strategy: "recursive",
  size: 512,
  overlap: 50,
  separator: "\n",
});

// Generate embeddings
const { embeddings } = await embedMany({
  model: openai.embedding("text-embedding-3-small"),
  values: chunks.map((chunk) => chunk.text),
});

// Get the vector store instance from Mastra
const vectorStore = mastra.getVector("pgVector");

// Create an index for our paper chunks
await vectorStore.createIndex({
  indexName: "papers",
  dimension: 1536,
});

// Store embeddings
await vectorStore.upsert({
  indexName: "papers",
  vectors: embeddings,
  metadata: chunks.map((chunk) => ({
    text: chunk.text,
    source: "transformer-paper",
  })),
});
