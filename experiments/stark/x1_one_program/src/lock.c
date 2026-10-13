/* Executable allowlist inside the one Python process (X1), applied before the interpreter runs any code.

   Loaded with LD_PRELOAD (from a sealed memory file) only by the gate's launch of the single granted
   interpreter. Its constructor adds a Landlock layer with the execute rules of the shell's allowlist layer
   (X1_LOCK_SPEC: lines "x<TAB>path" = execute+read, "r<TAB>path" = read), so the running interpreter cannot
   execute the Python binary again (nor awk, perl, node, ...). It then confirms on X1_LOCK_FD, closes the memory
   file and removes its variables from the environment.
   Any failure stops the process before Python starts (exit 126). Same access bits as the shell layer:
   handled = execute | read_file | read_dir. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#define FS_EXECUTE (1ULL << 0)
#define FS_READ_FILE (1ULL << 2)
#define FS_READ_DIR (1ULL << 3)

struct ruleset_attr {
    uint64_t handled_access_fs;
};
struct path_beneath_attr {
    uint64_t allowed_access;
    int32_t parent_fd;
} __attribute__((packed));

static void stop(const char *what) {
    char line[256];
    int n = snprintf(line, sizeof(line), "python3: interpreter setup failed (%s)\n", what);
    if (n > 0) {
        ssize_t ignored = write(2, line, (size_t)n);
        (void)ignored;
    }
    _exit(126);
}

__attribute__((constructor)) static void x1_lock(void) {
    const char *spec = getenv("X1_LOCK_SPEC");
    if (!spec) return;
    char *text = strdup(spec);
    if (!text) stop("memory");
    struct ruleset_attr attr = {FS_EXECUTE | FS_READ_FILE | FS_READ_DIR};
    long ruleset = syscall(SYS_landlock_create_ruleset, &attr, sizeof(attr), 0);
    if (ruleset < 0) stop("ruleset");
    char *save = NULL;
    for (char *line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        if (strlen(line) < 3 || line[1] != '\t') stop("rule format");
        char kind = line[0];
        const char *path = line + 2;
        int fd = open(path, O_PATH | O_CLOEXEC);
        if (fd < 0) {
            if (kind == 'r' && errno == ENOENT) continue;  /* optional readable path, as in the shell layer */
            stop("rule path");
        }
        struct stat st;
        if (fstat(fd, &st) != 0) stop("rule stat");
        uint64_t allowed;
        if (kind == 'x') allowed = FS_EXECUTE | FS_READ_FILE;
        else if (kind == 'r') allowed = FS_READ_FILE | (S_ISDIR(st.st_mode) ? FS_READ_DIR : 0);
        else stop("rule kind");
        struct path_beneath_attr rule = {allowed, fd};
        if (syscall(SYS_landlock_add_rule, (int)ruleset, 1, &rule, 0) != 0) stop("add rule");
        close(fd);
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) stop("no_new_privs");
    if (syscall(SYS_landlock_restrict_self, (int)ruleset, 0) != 0) stop("restrict");
    close((int)ruleset);
    free(text);
    const char *confirm = getenv("X1_LOCK_FD");
    if (confirm) {
        int fd = atoi(confirm);
        if (fd > 2) {
            if (write(fd, "L", 1) != 1) stop("confirm");
            close(fd);
        }
    }
    const char *memfd = getenv("X1_LOCK_MEMFD");  /* the sealed memory file this library was loaded from */
    if (memfd && atoi(memfd) > 2) close(atoi(memfd));
    unsetenv("X1_LOCK_SPEC");
    unsetenv("X1_LOCK_FD");
    unsetenv("X1_LOCK_MEMFD");
    unsetenv("LD_PRELOAD");
}
