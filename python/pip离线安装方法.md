## 方法1）快速下载命令（有网环境）
```bash
# 下载 pandas + 所有依赖到当前目录
pip config set global.cache-dir ~/.cache/pip
# 然后正常下载（已缓存的包会秒过）

pip download vllm --python-version 3.10 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --only-binary=:all: -d ./vllm_package

pip download openpyxl --python-version 3.10 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --only-binary=:all: -d ./package

```
离线安装命令（无网环境）
```bash
# 将下载的文件夹拷贝到离线机器，然后：
pip install --no-index --find-links=./pandas_packages pandas
```

## 方法2）利用requirements.txt离线安装依赖包
```
第一步：下载离线的包到指定目录
将requirements.txt中导入的包离线下载到 package_tmp_dir 文件夹
pip wheel -w package_tmp_dir -r requirements.txt 
或者
pip download -d package_tmp_dir -r requirements.txt     -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

第二步：安装离线的包
pip install --no-index --find-links=package_tmp_dir  -r requirements.txt

RUN pip install --find-links=package_tmp_dir_generate -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  # 同时找不到可以通过pipy源下载

```
