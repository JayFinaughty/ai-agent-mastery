import Langfuse from 'langfuse';

// Singleton Langfuse client instance
let langfuseClient: Langfuse | null = null;

/**
 * Get or create the Langfuse client for frontend feedback submission.
 * Returns null if Langfuse is not configured (graceful degradation).
 */
export function getLangfuseClient(): Langfuse | null {
  // Return cached instance if available
  if (langfuseClient !== null) {
    return langfuseClient;
  }

  const publicKey = import.meta.env.VITE_LANGFUSE_PUBLIC_KEY;
  const host = import.meta.env.VITE_LANGFUSE_HOST;

  // If not configured, return null (graceful degradation)
  if (!publicKey) {
    console.warn('[Langfuse] Not configured - feedback will not be submitted');
    return null;
  }

  langfuseClient = new Langfuse({
    publicKey,
    baseUrl: host || 'https://cloud.langfuse.com',
  });

  return langfuseClient;
}

/**
 * Submit a score to Langfuse. Non-blocking, fire-and-forget.
 * Handles missing trace_id or unconfigured Langfuse gracefully.
 */
export async function submitScore(params: {
  traceId: string | undefined;
  name: string;
  value: number;
  comment?: string;
}): Promise<boolean> {
  const { traceId, name, value, comment } = params;

  // Graceful degradation if trace_id is missing
  if (!traceId) {
    console.warn('[Langfuse] Cannot submit score - no trace_id');
    return false;
  }

  const client = getLangfuseClient();
  if (!client) {
    return false;
  }

  try {
    await client.score({
      traceId,
      name,
      value,
      comment,
    });
    return true;
  } catch (error) {
    console.error('[Langfuse] Failed to submit score:', error);
    return false;
  }
}
