// Import required modules
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleGenAI } from '@google/genai';
import { z } from "zod";
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Initialize Gemini API clients
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const genAINew = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Create MCP server instance
const server = new McpServer({
  name: "GeminiMediaServer",
  version: "2.0.0",
  description: "Server for generating images (Imagen) and videos (Veo 3) using Gemini API"
});

// ── Image Generation Tool ──────────────────────────────────────────────
server.tool(
  "generateImage",
  {
    prompt: z.string(),
    aspectRatio: z.string().optional(),
    outputFormat: z.string().optional()
  },
  async (params) => {
    const prompt = params.prompt || "Default image prompt";
    const aspectRatio = params.aspectRatio || "1:1";
    const outputFormat = params.outputFormat || "png";

    try {
      const model = genAI.getGenerativeModel({
        model: "gemini-2.0-flash-exp-image-generation",
        generationConfig: {
          responseModalities: ['Text', 'Image'],
        },
      });

      const result = await model.generateContent(prompt);

      // Collect all parts - prioritize image over text
      const parts = result.response.candidates[0].content.parts;
      let imageResult = null;
      let textResult = null;

      for (const part of parts) {
        if (part.inlineData) {
          imageResult = part.inlineData;
        } else if (part.text) {
          textResult = part.text;
        }
      }

      if (imageResult) {
        const imageData = imageResult.data;
        const filename = `gemini-image-${Date.now()}.${outputFormat}`;
        const buffer = Buffer.from(imageData, 'base64');
        fs.writeFileSync(filename, buffer);
        console.error(`Image saved as ${filename}`);
        return {
          content: [{
            type: "image",
            data: imageData,
            mimeType: `image/${outputFormat}`
          }]
        };
      } else if (textResult) {
        console.error(`No image generated, text response: ${textResult}`);
        return {
          content: [{
            type: "text",
            text: textResult
          }]
        };
      }
    } catch (error) {
      console.error("Image generation error:", error);
      return {
        content: [{
          type: "text",
          text: `Error generating image: ${error.message}`
        }]
      };
    }
  },
  {
    description: "Generate an image using Gemini Imagen",
    parameters: {
      prompt: { type: "string", description: "Text description of the image to generate" },
      aspectRatio: { type: "string", description: "Aspect ratio (e.g., '1:1', '4:3', '16:9')", optional: true },
      outputFormat: { type: "string", description: "Output format ('png' or 'jpeg')", optional: true }
    }
  }
);

// ── Video Generation Tool (Veo 3) ─────────────────────────────────────
server.tool(
  "generateVideo",
  {
    prompt: z.string(),
    aspectRatio: z.string().optional(),
    model: z.string().optional(),
    imagePath: z.string().optional(),
  },
  async (params) => {
    const prompt = params.prompt;
    const aspectRatio = params.aspectRatio || "9:16";
    const modelId = params.model || "veo-3.0-generate-001";

    try {
      console.error(`[Veo] Starting video generation: model=${modelId}, aspect=${aspectRatio}`);
      console.error(`[Veo] Prompt: ${prompt}`);

      // Build the request config
      const config = {
        aspectRatio: aspectRatio,
      };

      let operation;

      if (params.imagePath && fs.existsSync(params.imagePath)) {
        // Image-to-video: use provided image as first frame
        const imageBuffer = fs.readFileSync(params.imagePath);
        const imageBase64 = imageBuffer.toString('base64');
        const ext = path.extname(params.imagePath).toLowerCase();
        const mimeType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';

        console.error(`[Veo] Using image as first frame: ${params.imagePath}`);
        operation = await genAINew.models.generateVideos({
          model: modelId,
          prompt: prompt,
          image: {
            imageBytes: imageBase64,
            mimeType: mimeType,
          },
          config: config,
        });
      } else {
        // Text-to-video
        operation = await genAINew.models.generateVideos({
          model: modelId,
          prompt: prompt,
          config: config,
        });
      }

      console.error(`[Veo] Operation started, polling for completion...`);

      // Poll until done (check every 10 seconds, timeout after 5 minutes)
      const maxWait = 5 * 60 * 1000; // 5 minutes
      const pollInterval = 10 * 1000; // 10 seconds
      const startTime = Date.now();

      while (!operation.done) {
        const elapsed = Date.now() - startTime;
        if (elapsed > maxWait) {
          return {
            content: [{
              type: "text",
              text: `Video generation timed out after 5 minutes. Operation: ${operation.name}`
            }]
          };
        }
        console.error(`[Veo] Waiting... (${Math.round(elapsed / 1000)}s elapsed)`);
        await new Promise(resolve => setTimeout(resolve, pollInterval));
        operation = await genAINew.operations.getVideosOperation({ operation });
      }

      console.error(`[Veo] Generation complete!`);

      // Extract the video (API may use generatedSamples or generatedVideos)
      const samples = operation.response?.generatedSamples || operation.response?.generatedVideos || [];
      if (samples.length > 0) {
        const sample = samples[0];
        const videoUri = sample.video?.uri;

        if (videoUri) {
          // Download the video
          const separator = videoUri.includes('?') ? '&' : '?';
          const downloadUrl = `${videoUri}${separator}key=${process.env.GEMINI_API_KEY}`;
          console.error(`[Veo] Downloading video...`);
          const resp = await fetch(downloadUrl);

          if (!resp.ok) {
            return {
              content: [{
                type: "text",
                text: `Failed to download video: ${resp.status} ${resp.statusText}`
              }]
            };
          }

          const videoBuffer = Buffer.from(await resp.arrayBuffer());
          const filename = `gemini-video-${Date.now()}.mp4`;
          fs.writeFileSync(filename, videoBuffer);
          console.error(`[Veo] Video saved as ${filename} (${(videoBuffer.length / 1024 / 1024).toFixed(1)} MB)`);

          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                status: "success",
                file: path.resolve(filename),
                size_mb: (videoBuffer.length / 1024 / 1024).toFixed(1),
                model: modelId,
                aspectRatio: aspectRatio,
              })
            }]
          };
        }
      }

      return {
        content: [{
          type: "text",
          text: `Video generation completed but no video was returned. Response: ${JSON.stringify(operation.response)}`
        }]
      };

    } catch (error) {
      console.error("[Veo] Error:", error);
      return {
        content: [{
          type: "text",
          text: `Error generating video: ${error.message}`
        }]
      };
    }
  },
  {
    description: "Generate a video using Google Veo 3. Supports text-to-video and image-to-video. Async — may take 1-3 minutes.",
    parameters: {
      prompt: { type: "string", description: "Text description of the video to generate" },
      aspectRatio: { type: "string", description: "Aspect ratio: '9:16' (portrait, default) or '16:9' (landscape)", optional: true },
      model: { type: "string", description: "Model ID: 'veo-3.0-generate-001' (default) or 'veo-3.0-fast-generate-001' (faster/cheaper)", optional: true },
      imagePath: { type: "string", description: "Optional path to an image file to use as the first frame (image-to-video)", optional: true },
    }
  }
);

// Start the server
async function startServer() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("Gemini Media Server running (Imagen + Veo 3)");
  } catch (error) {
    console.error("Server startup error:", error);
  }
}

startServer();
