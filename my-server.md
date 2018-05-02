# My Server

```text
ssh terry@192.168.1.110
jupyter notebook --no-browser --port=8888
ssh -f terry@192.168.1.110 -L 8888:localhost:8888 -N

ssh -N -f -L localhost:8888:localhost:8888 terry@113.254.109.120 
sshfs terry@113.254.109.120: server 
```



