# Deep Learning for Computer Vision Adrian Rosebrock (STARTERS Bundle)
ShallowNet : 
              INPUT => CONV => RELU => FC

LeNet : 
              INPUT => CONV => RELU => POOL => CONV => RELU => POOL  => FC => RELU => FC

MiniVGGNet:  
              INPUT => ((CONV => RELU => BN) * 2 => POOL => DO) * 2  => FC => RELU => BN => DO => FC => SOFTMAX
