# AWS



## Seoul t2.micro 

```bash
ssh -i ~/.ssh/aws-key.pem ubuntu@ec2-13-124-252-134.ap-northeast-2.compute.amazonaws.com 
sudo sshfs {username}@{ipaddress}:{remote folder path}  {local folder path} -o IdentityFile={full path to the private key file} -o allow_other 
sudo sshfs ubuntu@ec2-13-124-252-134.ap-northeast-2.compute.amazonaws.com:/ ser-t2 -o IdentityFile=/home/terry/.ssh/aws-key.pem -o allow_other 
sudo sshfs -o allow_other,defer_permissions,IdentityFile=/home/terry/.ssh/aws-key.pem,loglevel=debug  ubuntu@ec2-13-124-252-134.ap-northeast-2.compute.amazonaws.com ser-t2 
scp -i ~/.ssh/aws-key.pem ~/porto/* ubuntu@ec2-13-124-252-134.ap-northeast-2.compute.amazonaws.com:~/porto/ 
```



{% embed data="{\"url\":\"https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html\",\"type\":\"link\",\"title\":\"Running Jupyter notebooks on GPU on AWS: a starter guide\",\"icon\":{\"type\":\"icon\",\"url\":\"https://blog.keras.io/favicon.ico\",\"aspectRatio\":0}}" %}



```bash
ssh -R 52698:localhost:52698 ubuntu@ec2-13-124-214-221.ap-northeast-2.compute.amazonaws.com 
pip install keras==1.2.2

```

* [\(guide\) Install Fastai in any AWS region – Pierre Guillou – Medium](https://medium.com/@pierre_guillou/guide-install-fastai-in-any-aws-region-8f4fe29132e5)

