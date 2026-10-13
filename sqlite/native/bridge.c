#define _GNU_SOURCE 1
#include "xu_native.h"
#include "../vendor/sqlite3.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

/* xu-c-1 exposes scalars only. Tokens own C resources; no pointer crosses ABI.
 * The registry lock also serializes close against every operation. */
typedef struct Object Object;
struct Object {
    xu_int id, owner;
    int kind, status, row;
    size_t length;
    void *ptr;
    Object *next;
};
enum { BUFFER = 1, DATABASE = 2, STATEMENT = 3, MAX_BUFFER = 16 * 1024 * 1024 };
static Object *objects;
static xu_int sequence;
#define EXPORT(name) XU_NATIVE_SYMBOL(xu_sqlite_native_##name)
#define ENTER sqlite3_initialize(); sqlite3_mutex *lock = sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1); sqlite3_mutex_enter(lock)
#define RETURN(value) do { __auto_type xu_bridge_return_value = (value); sqlite3_mutex_leave(lock); return xu_bridge_return_value; } while (0)

static Object *find(xu_int id, int kind) {
    for (Object *p = objects; p; p = p->next)
        if (p->id == id && p->kind == kind) return p;
    return NULL;
}
static Object *make(int kind, void *ptr) {
    if (sequence == LLONG_MAX) return NULL;
    Object *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->id = ++sequence; p->kind = kind; p->ptr = ptr;
    p->next = objects; objects = p;
    return p;
}
static void remove_object(Object *p) {
    Object **next = &objects;
    while (*next && *next != p) next = &(*next)->next;
    if (*next) *next = p->next;
    free(p);
}
static xu_int copy_buffer(const void *data, size_t length) {
    if (length > MAX_BUFFER || (!data && length)) return 0;
    unsigned char *bytes = calloc(length + 1, 1);
    if (!bytes) return 0;
    if (length) memcpy(bytes, data, length);
    Object *p = make(BUFFER, bytes);
    if (!p) { free(bytes); return 0; }
    p->length = length;
    return p->id;
}
static int cstring(Object *p) {
    return p && !memchr(p->ptr, 0, p->length);
}
static Object *statement(xu_int id) { return find(id, STATEMENT); }
static int record(Object *s, int rc) {
    if (s) {
        s->status = rc;
        Object *db = find(s->owner, DATABASE);
        if (db) db->status = rc;
    }
    return rc;
}

xu_int EXPORT(buffer_new)(xu_int length) {
    ENTER;
    if (length < 0 || length > MAX_BUFFER) RETURN(0);
    void *bytes = calloc((size_t)length + 1, 1);
    if (!bytes) RETURN(0);
    Object *p = make(BUFFER, bytes);
    if (!p) { free(bytes); RETURN(0); }
    p->length = (size_t)length;
    RETURN(p->id);
}
xu_int EXPORT(buffer_put)(xu_int id, xu_int offset, xu_int word, xu_int count) {
    ENTER;
    Object *p = find(id, BUFFER);
    if (!p || offset < 0 || count < 0 || count > 8 || (xu_uint)offset > p->length || (xu_uint)count > p->length - (size_t)offset) RETURN(SQLITE_MISUSE);
    xu_uint bits = (xu_uint)word;
    for (int i = 0; i < count; i++) ((unsigned char *)p->ptr)[offset + i] = (unsigned char)(bits >> (8 * i));
    RETURN(SQLITE_OK);
}
xu_int EXPORT(buffer_get)(xu_int id, xu_int offset) {
    ENTER;
    Object *p = find(id, BUFFER);
    if (!p || offset < 0 || (xu_uint)offset >= p->length) RETURN(0);
    xu_uint bits = 0;
    size_t count = p->length - (size_t)offset;
    if (count > 8) count = 8;
    for (size_t i = 0; i < count; i++) bits |= (xu_uint)((unsigned char *)p->ptr)[offset + i] << (8 * i);
    RETURN((xu_int)bits);
}
xu_int EXPORT(buffer_len)(xu_int id) {
    ENTER; Object *p = find(id, BUFFER); RETURN(p ? (xu_int)p->length : -1);
}
void EXPORT(buffer_free)(xu_int id) {
    ENTER; Object *p = find(id, BUFFER);
    if (p) { free(p->ptr); remove_object(p); }
    sqlite3_mutex_leave(lock);
}
xu_int EXPORT(open)(xu_int path) {
    ENTER; Object *b = find(path, BUFFER);
    if (!cstring(b)) RETURN(0);
    sqlite3 *db = NULL;
    int rc = sqlite3_open_v2(b->ptr, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI | SQLITE_OPEN_FULLMUTEX, NULL);
    if (!db) RETURN(0);
    Object *p = make(DATABASE, db);
    if (!p) { sqlite3_close_v2(db); RETURN(0); }
    p->status = rc;
    sqlite3_busy_timeout(db, 5000);
    RETURN(p->id);
}
xu_int EXPORT(status)(xu_int id) {
    ENTER; Object *p = find(id, DATABASE); RETURN(p ? p->status : SQLITE_MISUSE);
}
xu_int EXPORT(error)(xu_int id) {
    ENTER; Object *p = find(id, DATABASE);
    const char *s = p ? sqlite3_errmsg(p->ptr) : "invalid native database handle";
    if (p && p->status != SQLITE_OK && p->status != SQLITE_ROW && p->status != SQLITE_DONE && sqlite3_errcode(p->ptr) == SQLITE_OK) s = sqlite3_errstr(p->status);
    RETURN(copy_buffer(s, strlen(s)));
}
xu_int EXPORT(close)(xu_int id) {
    ENTER; Object *db = find(id, DATABASE);
    if (!db) RETURN(SQLITE_OK);
    Object *p = objects;
    while (p) {
        Object *next = p->next;
        if (p->kind == STATEMENT && p->owner == id) { sqlite3_finalize(p->ptr); remove_object(p); }
        p = next;
    }
    int rc = sqlite3_close_v2(db->ptr);
    if (rc == SQLITE_OK) remove_object(db);
    RETURN(rc);
}
xu_int EXPORT(prepare)(xu_int id, xu_int sql) {
    ENTER; Object *db = find(id, DATABASE), *text = find(sql, BUFFER);
    if (!db || !cstring(text)) { if (db) db->status = SQLITE_MISUSE; RETURN(0); }
    sqlite3_stmt *stmt = NULL;
    const char *tail = NULL;
    int rc = sqlite3_prepare_v2(db->ptr, text->ptr, (int)text->length + 1, &stmt, &tail);
    while (rc == SQLITE_OK && tail && *tail) {
        sqlite3_stmt *extra = NULL;
        const char *next = NULL;
        rc = sqlite3_prepare_v2(db->ptr, tail, -1, &extra, &next);
        if (extra) { sqlite3_finalize(extra); rc = SQLITE_MISUSE; }
        if (!next || next <= tail) break;
        tail = next;
    }
    db->status = rc;
    if (rc != SQLITE_OK) { sqlite3_finalize(stmt); RETURN(0); }
    if (!stmt) RETURN(0);
    Object *p = make(STATEMENT, stmt);
    if (!p) { sqlite3_finalize(stmt); db->status = SQLITE_NOMEM; RETURN(0); }
    p->owner = id;
    RETURN(p->id);
}
xu_int EXPORT(parameter_count)(xu_int id) {
    ENTER; Object *p = statement(id); RETURN(p ? sqlite3_bind_parameter_count(p->ptr) : -1);
}
xu_int EXPORT(parameter_name)(xu_int id, xu_int index) {
    ENTER; Object *p = statement(id);
    if (!p || index < 1 || index > sqlite3_bind_parameter_count(p->ptr)) RETURN(0);
    const char *s = sqlite3_bind_parameter_name(p->ptr, (int)index);
    RETURN(s ? copy_buffer(s, strlen(s)) : 0);
}
xu_int EXPORT(bind_null)(xu_int id, xu_int index) {
    ENTER; Object *p = statement(id); RETURN(p ? record(p, sqlite3_bind_null(p->ptr, (int)index)) : SQLITE_MISUSE);
}
xu_int EXPORT(bind_int)(xu_int id, xu_int index, xu_int value) {
    ENTER; Object *p = statement(id); RETURN(p ? record(p, sqlite3_bind_int64(p->ptr, (int)index, value)) : SQLITE_MISUSE);
}
/* Float text avoids relying on the current Xu native FP register adapter.
 * Formatting/parsing is locale independent and preserves 17-digit precision. */
xu_int EXPORT(bind_float)(xu_int id, xu_int index, xu_int buffer) {
    ENTER; Object *p = statement(id), *b = find(buffer, BUFFER);
    if (!p || !cstring(b)) RETURN(SQLITE_MISUSE);
    char *end = NULL;
#if defined(_WIN32)
    _locale_t locale = _create_locale(LC_NUMERIC, "C");
    if (!locale) RETURN(record(p, SQLITE_NOMEM));
    double value = _strtod_l(b->ptr, &end, locale);
    _free_locale(locale);
#else
    locale_t locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    if (!locale) RETURN(record(p, SQLITE_NOMEM));
    double value = strtod_l(b->ptr, &end, locale);
    freelocale(locale);
#endif
    if (end == (char *)b->ptr || end != (char *)b->ptr + b->length) RETURN(record(p, SQLITE_MISMATCH));
    RETURN(record(p, sqlite3_bind_double(p->ptr, (int)index, value)));
}
xu_int EXPORT(bind_buffer)(xu_int id, xu_int index, xu_int buffer, xu_int blob) {
    ENTER; Object *p = statement(id), *b = find(buffer, BUFFER);
    if (!p || !b) RETURN(SQLITE_MISUSE);
    int rc = blob ? sqlite3_bind_blob(p->ptr, (int)index, b->ptr, (int)b->length, SQLITE_TRANSIENT)
                  : sqlite3_bind_text(p->ptr, (int)index, b->ptr, (int)b->length, SQLITE_TRANSIENT);
    RETURN(record(p, rc));
}
xu_int EXPORT(step)(xu_int id) {
    ENTER; Object *p = statement(id);
    if (!p) RETURN(SQLITE_MISUSE);
    int rc = record(p, sqlite3_step(p->ptr)); p->row = rc == SQLITE_ROW; RETURN(rc);
}
void EXPORT(finalize)(xu_int id) {
    ENTER; Object *p = statement(id);
    if (p) { sqlite3_finalize(p->ptr); remove_object(p); }
    sqlite3_mutex_leave(lock);
}
static int column_ok(Object *p, xu_int column, int row) {
    return p && (!row || p->row) && column >= 0 && column < sqlite3_column_count(p->ptr);
}
xu_int EXPORT(column_count)(xu_int id) {
    ENTER; Object *p = statement(id); RETURN(p ? sqlite3_column_count(p->ptr) : -1);
}
xu_int EXPORT(column_name)(xu_int id, xu_int column) {
    ENTER; Object *p = statement(id);
    if (!column_ok(p, column, 0)) RETURN(0);
    const char *s = sqlite3_column_name(p->ptr, (int)column);
    RETURN(s ? copy_buffer(s, strlen(s)) : 0);
}
xu_int EXPORT(column_type)(xu_int id, xu_int column) {
    ENTER; Object *p = statement(id); RETURN(column_ok(p, column, 1) ? sqlite3_column_type(p->ptr, (int)column) : 0);
}
xu_int EXPORT(column_int)(xu_int id, xu_int column) {
    ENTER; Object *p = statement(id); RETURN(column_ok(p, column, 1) ? sqlite3_column_int64(p->ptr, (int)column) : 0);
}
xu_int EXPORT(column_float)(xu_int id, xu_int column) {
    ENTER; Object *p = statement(id);
    if (!column_ok(p, column, 1)) RETURN(0);
    char *text = sqlite3_mprintf("%!.17g", sqlite3_column_double(p->ptr, (int)column));
    if (!text) RETURN(0);
    xu_int result = copy_buffer(text, strlen(text));
    sqlite3_free(text);
    RETURN(result);
}
xu_int EXPORT(column_buffer)(xu_int id, xu_int column, xu_int blob) {
    ENTER; Object *p = statement(id);
    if (!column_ok(p, column, 1)) RETURN(0);
    const void *data = blob ? sqlite3_column_blob(p->ptr, (int)column) : sqlite3_column_text(p->ptr, (int)column);
    int length = sqlite3_column_bytes(p->ptr, (int)column);
    if (!data && sqlite3_errcode(sqlite3_db_handle(p->ptr)) == SQLITE_NOMEM) RETURN(0);
    RETURN(copy_buffer(data, (size_t)length));
}
xu_int EXPORT(last_id)(xu_int id) {
    ENTER; Object *p = find(id, DATABASE); RETURN(p ? sqlite3_last_insert_rowid(p->ptr) : 0);
}
xu_int EXPORT(total_changes)(xu_int id) {
    ENTER; Object *p = find(id, DATABASE); RETURN(p ? sqlite3_total_changes64(p->ptr) : 0);
}

xu_int EXPORT(changes)(xu_int id) {
    ENTER; Object *p = find(id, DATABASE); RETURN(p ? sqlite3_changes64(p->ptr) : 0);
}
