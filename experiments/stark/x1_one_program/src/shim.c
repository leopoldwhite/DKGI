/* python3 for the One program condition (X1).

   This program is installed as `python3` in the sandbox PATH. It never starts an interpreter itself. It hands
   its standard streams and working directory to the per-command gate (inherited descriptor 198, outside the
   executable allowlist), which asks the harness-held counter for the single start. Refused or unreachable:
   print the fixed refusal and exit 126 without starting an interpreter. Granted: wait for the interpreter
   that the gate started on these streams and exit with its status (or die by the same signal). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#define GATE_FD 198
static const char REFUSAL[] = "python3: unavailable: Python 3 can be started only once in this task.\n";
extern char **environ;

static int refuse(void) {
    ssize_t ignored = write(2, REFUSAL, sizeof(REFUSAL) - 1);
    (void)ignored;
    return 126;
}

static int send_all(int fd, const char *data, size_t n) {
    while (n) {
        ssize_t k = write(fd, data, n);  /* write(2): the sandbox's seccomp filter refuses send/recv */
        if (k < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        data += k;
        n -= (size_t)k;
    }
    return 0;
}

static int recv_all(int fd, char *data, size_t n) {
    while (n) {
        ssize_t k = read(fd, data, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        data += k;
        n -= (size_t)k;
    }
    return 0;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct stat st;
    if (fstat(GATE_FD, &st) != 0 || !S_ISSOCK(st.st_mode)) return refuse();
    for (int fd = 0; fd < 3; fd++) {
        if (fcntl(fd, F_GETFD) == -1) {
            int n = open("/dev/null", O_RDWR);
            if (n >= 0 && n != fd) {
                dup2(n, fd);
                close(n);
            }
        }
    }
    int cwd = open(".", O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (cwd < 0) cwd = open("/", O_PATH | O_DIRECTORY | O_CLOEXEC);
    int pair[2];
    if (cwd < 0 || socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) != 0) return refuse();

    int fds[5] = {pair[1], 0, 1, 2, cwd};
    char header[4] = {'X', '1', 'L', '1'};
    struct iovec iov = {header, sizeof(header)};
    union {
        char buf[CMSG_SPACE(sizeof(fds))];
        struct cmsghdr align;
    } control;
    memset(&control, 0, sizeof(control));
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.buf;
    msg.msg_controllen = sizeof(control.buf);
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(fds));
    memcpy(CMSG_DATA(cm), fds, sizeof(fds));
    if (sendmsg(GATE_FD, &msg, MSG_NOSIGNAL) != (ssize_t)sizeof(header)) return refuse();
    close(pair[1]);
    close(cwd);

    /* Payload: u32 argc, u32 envc, u32 bytes, then argc + envc NUL-terminated strings. */
    uint32_t envc = 0;
    size_t bytes = 0;
    for (int i = 0; i < argc; i++) bytes += strlen(argv[i]) + 1;
    for (char **e = environ; *e; e++, envc++) bytes += strlen(*e) + 1;
    if (bytes > (64u << 20)) return refuse();
    char *blob = malloc(bytes + 12);
    if (!blob) return refuse();
    uint32_t head[3] = {(uint32_t)argc, envc, (uint32_t)bytes};
    memcpy(blob, head, 12);
    char *p = blob + 12;
    for (int i = 0; i < argc; i++) {
        size_t n = strlen(argv[i]) + 1;
        memcpy(p, argv[i], n);
        p += n;
    }
    for (char **e = environ; *e; e++) {
        size_t n = strlen(*e) + 1;
        memcpy(p, *e, n);
        p += n;
    }
    if (send_all(pair[0], blob, bytes + 12) != 0) return refuse();
    free(blob);

    char kind;
    if (recv_all(pair[0], &kind, 1) != 0 || kind != 'G') return refuse();
    char status[5];
    if (recv_all(pair[0], status, 5) != 0) return 1;  /* the interpreter's supervisor stopped */
    int32_t value;
    memcpy(&value, status + 1, 4);
    if (status[0] == 'E') return value & 0xff;
    if (status[0] == 'S' && value > 0 && value < 65) {
        signal(value, SIG_DFL);
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, value);
        sigprocmask(SIG_UNBLOCK, &set, NULL);
        raise(value);
        return 128 + value;
    }
    return 126;  /* 'F': the gate could not start the interpreter */
}
