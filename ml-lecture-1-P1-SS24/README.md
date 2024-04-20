# Getting started with the Machine Learning lecture's programming exercises

1. Install Python. This is usually only necessary on Windows, you can download Pythonm here: https://www.python.org/downloads/. The scripts have been tested with Python 3.10 on Windows, so we recommend this version. However, version 3.8 and 3.9 should also work, and of course you are allowed to use Linux or Mac as well. 
Note, however, that we assume you are a student of a subject closely related to Computer Science. Therefore, you should be proficient with setting up your operating system and the tools needed yourself. Please understand that we cannot give any assistance with OS-related issues, so you should first ask your fellow students for help with such technical issues.   

2. Install a Python programming environment. We highly recommend PyCharm, which you can download here: https://www.jetbrains.com/de-de/pycharm/download/
The Community Edition should be sufficient for our purposes, but if you want, you can also get the Professional Edition if you register with your student email address. 

4. Set up a virtual Python interpreter. In PyCharm, you can do so by clicking "File->Settings->Project:ml-lecture->Python Interpreter->|Gear Symbol in upper right corner|->Add" and then make sure "Virtualenv Environment" is checked and "New Environment" is selected. Also, you have to select your previously installed Python interpreter in "Base Interpreter".

5. Install the python requirements. The easiest way of installing them is to use PyCharm. In PyCharm, when you open any python file, there will appear a warning about missing packages in a yellow banner at the top of the editor. Here, you can simply click on "Install Requirements". 
 Another way of installing the requirements **in Linux only(!!!)** is by clicking "Terminal" at the bottom of your Pycharm window, and, after you have set up your Python Interpreter, execute the command "pip install -r requirements.txt"

6. For each exercise, create a debug configuration. For example, for the 2d2nd exercise, open the file ``2d2nd.py`` and click "Run->Debug...", then, in the mini-window that will open, click "0. Edit configuration" .In the config window that opens then, make sure the file ``2d2nd.py`` is selected under "script path" and in "Python interpreter" the virtualenv interpreter you have set up previously is chosen. 

7. If you use PyCharm you have to enable drawing of figures in an own window for better visualization. To do this, click "File->Settings->Tools->Python Scientific" and unselect "Show plots in tool window". 