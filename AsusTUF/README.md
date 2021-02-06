# System Info
2021 Asus TUF Gaming A15
CPU: AMD Ryzen7 5800H
GPU: GeForce RTX 3070 Laptop
Memory: 16GB

# NVIDIA
driver, cudatoolkit, cudnn.

* Driver
  Graphical install, successfully installed NVIDIA Driver.
  But in reboot, error reported: usci_acpi USBC000:00 PPM init failed (-110)
  Some explanation: 
    UCSI = "USB Type-C Connector System Software Interface"
    PPM = "Platform Policy Manager. Hardware/firmware that manages all the USB Type-C connectors on the platform."
  Conflicts between NVIDIA 460.39 driver and Looks like USB-C port???
  Tried method:
    Update Ubuntu kernel to 5.10.0 -> same issue
    Update Ubuntu kernel to 5.11.0 -> cannot boot
    Use a lower version driver -> nvidia-smi Could not find devices, nvcc -V has output.
