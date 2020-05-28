## Rasperry Pi Mask Detection

Deploying the mask detection project is very easy. 
1. SCP _rpi_object_detection_v3.py_ AND MODEL AND PROTOTXXT to your Raspberry Pi. 
2. Connect a DSLR connected to your Rpi.
3. Ensure gphoto2 is properly installed http://www.gphoto.org/. This will control your DSLR from the Rpi
4. Run rpi_object_detection_v3. Ensure the proper args are in place
5. At the end of each day, transfer assets from your camera to your main workstation. Assets are to be placed in a folder for the current date and then processed using _data_processor.py_


### Improvements
It would be preferable to automatically transfer assets from the Rpi / DSLR to your main workstation. I had to do many things localy, so that piece is not built out yet  
 

 

 