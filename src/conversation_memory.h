#ifndef CONVERSATION_MEMORY_H
#define CONVERSATION_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ConversationMemory ConversationMemory;

typedef struct {
    const char *role;     /* "user" or "assistant" */
    const char *content;  /* message text */
    double timestamp;     /* Unix timestamp */
} MemoryTurn;

/**
 * Create conversation memory.
 * @param path         File path for persistence (e.g., "~/.pocket-voice/history.jsonl")
 * @param max_turns    Maximum turns to keep in memory (default: 50)
 * @param max_tokens   Approximate max token count for context window (default: 4000)
 * @return             Handle, or NULL on failure
 */
ConversationMemory *memory_create(const char *path, int max_turns, int max_tokens);

/** Destroy and free. Safe to call with NULL. */
void memory_destroy(ConversationMemory *mem);

/**
 * Add a turn to memory. Automatically persists to disk.
 * @param mem     Handle
 * @param role    "user" or "assistant"
 * @param content Message text
 * @return        0 on success, -1 on error
 */
int memory_add_turn(ConversationMemory *mem, const char *role, const char *content);

/**
 * Get recent turns for LLM context.
 * Returns turns that fit within max_tokens, most recent first.
 * @param mem       Handle
 * @param turns_out Array to fill (caller-allocated)
 * @param max_out   Size of turns_out array
 * @return          Number of turns returned
 */
int memory_get_context(ConversationMemory *mem, MemoryTurn *turns_out, int max_out);

/**
 * Format context as a single string for LLM prompt.
 * Returns malloc'd string like:
 *   "Previous conversation:\nUser: Hello\nAssistant: Hi there!\n..."
 * Caller must free.
 */
char *memory_format_context(ConversationMemory *mem);

/** Get total number of turns stored. */
int memory_turn_count(const ConversationMemory *mem);

/** Clear all turns (both in-memory and on disk). */
void memory_clear(ConversationMemory *mem);

/**
 * Search turns for a keyword.
 * Returns turns containing the keyword (case-insensitive).
 * @param mem       Handle
 * @param keyword   Search string
 * @param turns_out Array to fill
 * @param max_out   Size of turns_out
 * @return          Number of matching turns
 */
int memory_search(ConversationMemory *mem, const char *keyword,
                   MemoryTurn *turns_out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* CONVERSATION_MEMORY_H */
