#!/usr/bin/env python
# coding: utf-8

import struct
import pickle



def client_send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def server_send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send


def client_recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg = recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg


def server_recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg = recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen


def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''

    # packet = sock.recv(min(n - len(data), 2048))
    # if packet == b'':
    # raise RuntimeError("socket connection broken")

    while len(data) < n:

        packet = sock.recv(n - len(data))

        if not packet:
            socket_status = is_remote_alive(sock)
            print()
            print("Received null value. Reason: Socket is " + str(socket_status))
            print("Check your client log(s) to get clue.")
            print()
            return None
        data += packet
    return data


from ctypes import (
    CDLL, c_int, POINTER, Structure, c_void_p, c_size_t,
    c_short, c_ssize_t, c_char, ARRAY
)

__all__ = 'is_remote_alive',


class pollfd(Structure):
    _fields_ = (
        ('fd', c_int),
        ('events', c_short),
        ('revents', c_short),
    )


MSG_DONTWAIT = 0x40
MSG_PEEK = 0x02

EPOLLIN = 0x001
EPOLLPRI = 0x002
EPOLLRDNORM = 0x040

libc = CDLL(None)

recv = libc.recv
recv.restype = c_ssize_t
recv.argtypes = c_int, c_void_p, c_size_t, c_int

poll = libc.poll
poll.restype = c_int
poll.argtypes = POINTER(pollfd), c_int, c_int


class IsRemoteAlive:  # not needed, only for debugging
    def __init__(self, alive, msg):
        self.alive = alive
        self.msg = msg

    def __str__(self):
        return self.msg

    def __repr__(self):
        return 'IsRemoteClosed(%r,%r)' % (self.alive, self.msg)

    def __bool__(self):
        return self.alive


def is_remote_alive(fd):
    fileno = getattr(fd, 'fileno', None)
    if fileno is not None:
        if hasattr(fileno, '__call__'):
            fd = fileno()
        else:
            fd = fileno

    p = pollfd(fd=fd, events=EPOLLIN | EPOLLPRI | EPOLLRDNORM, revents=0)
    result = poll(p, 1, 0)
    if not result:
        return IsRemoteAlive(True, 'empty')

    buf = ARRAY(c_char, 1)()
    result = recv(fd, buf, len(buf), MSG_DONTWAIT | MSG_PEEK)
    if result > 0:
        return IsRemoteAlive(True, 'readable')
    elif result == 0:
        return IsRemoteAlive(False, 'closed')
    else:
        return IsRemoteAlive(False, 'errored')
