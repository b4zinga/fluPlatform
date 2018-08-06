#!/usr/bin/env python
# coding: utf-8
# Date  : 2018-08-06 11:36:15
# Author: b4zinga
# Email : b4zinga@outlook.com
# Func  : Main

import sys

PyVersion = sys.version.split()[0]

if __name__ == '__main__':
    if PyVersion <= "3":
        print("[-] For successfully running, you'll have to use python version 3.5 or later.")
        exit(0)

    import showdata
    app = showdata.Application()
    app.addFunc(showdata.FunctionMenu)
    app.mainloop()
