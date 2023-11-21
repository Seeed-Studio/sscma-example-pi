## When using a csi camera as input, it is necessary to confirm whether the camera is open or not

   ```bash
   sudo raspi-config
   ```

   select Interfacing Options -> Camera -> Yes -> OK -> Finish -> Yes -> OK -> Reboot Now

   If there is no camera option in raspi-config, you can comment it out in /boot/config.txt
   ```bash
   camera-auto-detect=0
   ```
   After restarting, run the following command to check whether the camera is on
   ```bash
   vcgencmd get_camera
   ```

   If supported=1 detected=1 then the camera is on, otherwise try again

   View camera device number
   ```bash
    v4l2-ctl --list-devices
   ```
   Change the device number of the device in run.sh
   ```bash
    device=/dev/videoX
    ```
