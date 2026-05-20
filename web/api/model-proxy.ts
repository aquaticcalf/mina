export const config = {
  runtime: 'edge',
};

// Allowed ONNX files
const ALLOWLIST = ['best.onnx', 'fish_gate.onnx'];

export default async function handler(req: Request) {
  try {
    const url = new URL(req.url);
    const file = url.searchParams.get('file');

    if (!file || !ALLOWLIST.includes(file)) {
      return new Response('Invalid file requested', { status: 400 });
    }

    const githubUrl = `https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/${file}`;
    const response = await fetch(githubUrl);

    if (!response.ok) {
      return new Response(`Failed to fetch from GitHub: ${response.status}`, {
        status: response.status,
      });
    }

    // Proxy the response, streaming the body to save memory on edge
    return new Response(response.body, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        // Important: CORS headers
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        // Cache heavily at the edge, as these files are immutable per release
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (error) {
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
}
