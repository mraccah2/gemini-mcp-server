// Import required modules
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { GoogleGenAI } from '@google/genai';
import { z } from "zod";
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Initialize Gemini API
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Create MCP server instance
const server = new McpServer({
  name: "GeminiMediaServer",
  version: "2.0.0",
  description: "Server for generating images (Imagen) and videos (Veo 3.1) using Gemini API"
});

// Image generation tool — collects all parts, prioritizes image over text
server.tool(
  "generateImage",
  {
    prompt: z.string(),
    aspectRatio: z.string().optional(),
    outputFormat: z.string().optional()
  },
  async (params) => {
    const prompt = params.prompt || "Default image prompt";
    const outputFormat = params.outputFormat || "png";

    try {
      const response = await ai.models.generateContent({
        model: "gemini-2.0-flash-exp-image-generation",
        contents: prompt,
        config: {
          responseModalities: ["Text", "Image"],
        },
      });

      // Collect all parts — prioritize image over text
      let imagePart = null;
      let textPart = null;

      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData) {
          imagePart = part;
        } else if (part.text) {
          textPart = part;
        }
      }

      if (imagePart) {
        const imageData = imagePart.inlineData.data;
        const buffer = Buffer.from(imageData, 'base64');
        const timestamp = Date.now();
        const filePath = `/tmp/claude/gemini-image-${timestamp}.${outputFormat}`;
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        fs.writeFileSync(filePath, buffer);

        return {
          content: [
            {
              type: "text",
              text: `Image saved to: ${filePath}`
            },
            {
              type: "image",
              data: imageData,
              mimeType: `image/${outputFormat}`
            }
          ]
        };
      } else if (textPart) {
        return {
          content: [{
            type: "text",
            text: `Gemini returned text instead of image: ${textPart.text}`
          }]
        };
      } else {
        return {
          content: [{
            type: "text",
            text: "No image or text returned by Gemini."
          }]
        };
      }
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: `Error generating image: ${error.message}`
        }]
      };
    }
  },
  {
    description: "Generate an image using Gemini Imagen. Returns the image and saves it locally.",
    parameters: {
      prompt: { type: "string", description: "Text description of the image to generate" },
      aspectRatio: { type: "string", description: "Aspect ratio (e.g., '1:1', '16:9', '9:16')", optional: true },
      outputFormat: { type: "string", description: "Output format ('png' or 'jpeg')", optional: true }
    }
  }
);

// Video generation tool — Google Veo 3.1 with native audio via Gemini API
server.tool(
  "generateVideo",
  {
    prompt: z.string(),
    imagePath: z.string().optional(),
    aspectRatio: z.string().optional(),
    duration: z.string().optional()
  },
  async (params) => {
    const prompt = params.prompt;
    const durationStr = params.duration || "8s";
    const durationNum = parseInt(durationStr.replace("s", ""), 10) || 8;
    // Veo 3.1 supports 4, 6, or 8 seconds
    const validDuration = [4, 6, 8].includes(durationNum) ? durationNum : 8;
    const aspectRatio = params.aspectRatio || "9:16";

    try {
      // Build generate request config
      const generateConfig = {
        aspectRatio: aspectRatio,
      };

      // If image provided, use image-to-video
      let image = undefined;
      if (params.imagePath) {
        const imageBuffer = fs.readFileSync(params.imagePath);
        const ext = path.extname(params.imagePath).slice(1).toLowerCase();
        const mimeType = ext === 'jpg' ? 'image/jpeg' : `image/${ext}`;
        image = {
          imageBytes: imageBuffer.toString('base64'),
          mimeType: mimeType,
        };
      }

      // Start video generation using the SDK
      let operation = await ai.models.generateVideos({
        model: "veo-3.1-generate-preview",
        prompt: prompt,
        image: image,
        config: generateConfig,
      });

      // Poll until complete (10s intervals, 6 min timeout)
      const maxWaitMs = 6 * 60 * 1000;
      const pollIntervalMs = 10_000;
      const startTime = Date.now();

      while (!operation.done) {
        if (Date.now() - startTime > maxWaitMs) {
          return {
            content: [{
              type: "text",
              text: `Video generation timed out after 6 minutes. Operation: ${operation.name}`
            }]
          };
        }

        await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
        operation = await ai.operations.getVideosOperation({ operation: operation });
      }

      // Check for errors
      if (operation.error) {
        return {
          content: [{
            type: "text",
            text: `Veo 3.1 video generation failed: ${JSON.stringify(operation.error)}`
          }]
        };
      }

      // Download generated video(s)
      const generatedVideos = operation.response?.generatedVideos || [];
      if (generatedVideos.length === 0) {
        return {
          content: [{
            type: "text",
            text: `Veo 3.1 completed but no videos in response. Operation: ${JSON.stringify(operation)}`
          }]
        };
      }

      const video = generatedVideos[0].video;
      const timestamp = Date.now();
      const filePath = `/tmp/claude/veo-video-${timestamp}.mp4`;
      fs.mkdirSync(path.dirname(filePath), { recursive: true });

      // Download using the SDK — try both method names for compatibility
      if (typeof ai.files.downloadFile === 'function') {
        await ai.files.downloadFile({ file: video, downloadPath: filePath });
      } else if (typeof ai.files.download === 'function') {
        await ai.files.download({ file: video, downloadPath: filePath });
      } else {
        // Fallback: download via REST API
        const apiKey = process.env.GEMINI_API_KEY;
        const videoUri = video.uri || video;
        const downloadUrl = typeof videoUri === 'string' && videoUri.startsWith('http')
          ? videoUri
          : `https://generativelanguage.googleapis.com/v1beta/${videoUri}?key=${apiKey}&alt=media`;
        const videoResp = await fetch(downloadUrl);
        if (!videoResp.ok) {
          return {
            content: [{
              type: "text",
              text: `Failed to download video: HTTP ${videoResp.status}. URI: ${videoUri}`
            }]
          };
        }
        const videoBuffer = Buffer.from(await videoResp.arrayBuffer());
        fs.writeFileSync(filePath, videoBuffer);
      }

      const fileSize = fs.statSync(filePath).size;
      return {
        content: [{
          type: "text",
          text: `Video saved to: ${filePath} (${(fileSize / 1024 / 1024).toFixed(1)} MB)`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: `Error generating video: ${error.message}`
        }]
      };
    }
  },
  {
    description: "Generate a video with native audio using Google Veo 3.1. Supports dialogue, sound effects, ambient audio with lip sync. Duration: 4, 6, or 8 seconds.",
    parameters: {
      prompt: { type: "string", description: "Text description of the video to generate. Include dialogue in quotes for narration with lip sync." },
      imagePath: { type: "string", description: "Optional local path to an image for image-to-video generation", optional: true },
      aspectRatio: { type: "string", description: "Aspect ratio ('9:16' or '16:9'). Default: '9:16'", optional: true },
      duration: { type: "string", description: "Video duration ('4s', '6s', or '8s'). Default: '8s'", optional: true }
    }
  }
);

// Start the server
async function startServer() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
  } catch (error) {
    console.error("Server startup error:", error);
  }
}

startServer();
