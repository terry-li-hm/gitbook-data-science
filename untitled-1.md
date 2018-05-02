# Commands



```bash
watch -n5 nvidia-smi


# read names 
# split_file = CARVANA_DIR +'/split/'+ split 
split_file = 'split/'+ split 
with open(split_file) as f: 
        names = f.readlines() 
names = [name.strip()for name in names] 
num   = len(names) 


$ cd ~/opencv-3.3.0/ 
$ mkdir build 
$ cd build 
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \ 
    -D CMAKE_INSTALL_PREFIX=/usr/local \ 
    -D INSTALL_PYTHON_EXAMPLES=ON \ 
    -D INSTALL_C_EXAMPLES=OFF \ 
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \ 
    -D PYTHON_EXECUTABLE=~/usr/bin/python \ 
    -D BUILD_EXAMPLES=ON .. 


|& tee -a ~/log.txt 


ssh -N -f -L localhost:8888:localhost:8888 terry@113.254.109.4
sshfs terry@113.254.109.4: server
sshfs terry@113.254.109.4:/3t 3t



```

