import sys

"""
@package Colored printing functions for strings that use universal ANSI escape sequences.
"""


class Base:
    # Foreground:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # End colored text
    END = '\033[0m'
    NC = '\x1b[0m'  # No Color


def print_fail(*messages, end='\n'):
    for message in messages:
        print(Base.BOLD + Base.FAIL + str(message) + Base.END, end="", sep=" ")
    print(end=end)
    sys.exit(0)


def print_pass(*messages, end='\n'):
    for message in messages:
        print(Base.BOLD + Base.OKGREEN + str(message) + Base.END, end="", sep=" ")
    print(end=end)


def print_warn(*messages, end='\n'):
    for message in messages:
        print(Base.BOLD + Base.WARNING + str(message) + Base.END, end="", sep=" ")
    print(end=end)


def print_notify(*messages, end='\n'):
    for message in messages:
        print(Base.BOLD + Base.OKBLUE + str(message) + Base.END, end="", sep=" ")
    print(end=end)


def print_info(*messages, end='\n'):
    for message in messages:
        print(message, end="", sep=" ")
    print(end=end)


def print_bold(*messages, end='\n'):
    for message in messages:
        print(Base.BOLD + str(message) + Base.END, end="", sep=" ")
    print(end=end)
