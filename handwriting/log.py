from colorama import init, Fore, Back, Style



def print_red(info, value=""):
    """Utility to print a message in red

    """

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


