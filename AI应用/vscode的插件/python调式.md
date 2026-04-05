一） 创建python的默认目录
安装debug的插件python debugger

![[AI应用/vscode的插件/assets/python调式/f8dd3af9149faf4f552640b5ffeb3bf3_MD5.png]]

![[AI应用/vscode的插件/assets/python调式/eb532336edf493618a8ab5c04530a23d_MD5.png]]

![[AI应用/vscode的插件/assets/python调式/88b21fa6c7f7750c534dff367764006e_MD5.png]]

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}"
        }
    ]
}