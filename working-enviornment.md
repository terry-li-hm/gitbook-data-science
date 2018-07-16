# Working Enviornment

* [neilpanchal/spinzero-jupyter-theme: A minimal Jupyter Notebook theme](https://github.com/neilpanchal/spinzero-jupyter-theme)
* [http://wpad/wpad.dat](http://wpad/wpad.dat) \(to find proxy\)
* [agermanidis/livepython: Visually trace Python code in real-time.](https://github.com/agermanidis/livepython)
* [Home - Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
* [如何与深度学习服务器优雅的交互？（长期更新）](https://zhuanlan.zhihu.com/p/32496193)
* [Useful snippets for Google Colaboratory. Free GPU included.](https://medium.com/machine-learning-world/useful-snippets-for-google-colaboratory-free-gpu-included-d976d6b3e6de)
* [Automating Windows Applications Using COM - Practical Business Python](http://pbpython.com/windows-com.html)
* [jupyter/repo2docker: Turn git repositories into Jupyter enabled Docker Images](https://github.com/jupyter/repo2docker)

```bash
watch -n5 nvidia-smi
```

## Windows

Remote access Jupyter notebook from Windows

> 1. Download the latest version of [PUTTY](http://www.putty.org/)
> 2. Open PUTTY and enter the server URL or IP address as the hostname
> 3. Now, go to SSH on the bottom of the left pane to expand the menu and then click on Tunnels
> 4. Enter the port number which you want to use to access Jupyter on your local machine. Choose 8000 or greater \(ie 8001, 8002, etc.\) to avoid ports used by other services, and set the destination as localhost:8888 where :8888 is the number of the port that Jupyter Notebook is running on. Now click the Add button, and the ports should appear in the Forwarded ports list.
> 5. Finally, click the Open button to connect to the server via SSH and tunnel the desired ports. Navigate to [http://localhost:8000](http://localhost:8000/) \(or whatever port you chose\) in a web browser to connect to Jupyter Notebook running on the server.

