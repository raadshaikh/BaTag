# Description

This project involves the redesign of a camera module for 
cryogenic vacuum vessel observation.

The designer of the original module passed away and can not 
provide exact information for replacement electronic parts.

Troubleshooting the original design indicated electrical 
failure of both the USB extender module and Logitech webcam 
module likely due to a power failure.

# Replacement Camera Search

The requirements for the new camera are as follow:

- No IR LEDs built into the camera assembly. 
  - These LEDs are uncommon outside of surveillance cameras 
    and, chances are, we will not be able to control these 
    LEDs on most cameras we take apart. 
    An externally controllable LED ring is recommended. 
    Hopefully this camera design can be reused and 
    always-on/uncontrolled IR are expected to be a problem 
    for some projects. 
    The existing LED ring can be reused and an IR version 
    built if needed.
- Fixed lens
    - Likely insufficient space in chassis for lens motors
- 2.8mm (among market options) focal length for largest FOV
- No POE, 12VDC power instead. 
  - We have extra pins on the connectors so I think using 
    POE is an unnecessary extra cost.
- Should fit in existing vacuum vessel chassis to avoid 
  mechanical redesign
- As previously discussed, interface should be Ethernet 
  based due to low bandwidth requirements and the vacuum 
  feedthgough being a standard D-Sub connector.
- 4K for good digital zoom abilities.
- Sony sensor preferred. Sony's sensors tend to have 
  outstanding low light performance and are prevalent in 
  the market. This may substitute the IR LEDs
- No disassembly required.
- Good brand reputation for long life and possible support.
  
Some options were found:

#. [IMX335 module][Ali_IP_cam1]
#. [IMX415 module][Ali_IP_cam2]
#. [Fisheye camera][Ali_IP_cam3]
#. [REVODATA I704-P][REVO_I704_P]
   - This camera initially started the discussion about IR LEDs
   - [REVODATA seems to be an OEM][REVODATA] of general 
     security cameras but this model is not listed on their 
     website
#. [BlueFishCam 5MP POE][BlueFishCam_5MP]
   - [Ansice is the suspected OEM][Ansice_M4MN41]
#. [NC 8MP][Ansice_H8M415_Amazon]
   - [Ansice is the suspected OEM][Ansice_H8M415]
#. [Ansice M8MN82][Ansice_M8MN82_Amazon]
   - [Ansice is the OEM][Ansice_M8MN82]
   - Ansice part numbering is very confusing. 
     M8MN82 is from the web page title.
     Alternate numbers include 
     "Camera Model" (on description page): B604, 
     "Model Number" (in datasheet): H8MF15
#. [Ansice A8M415][Ansice_A8M415]
  - Ansice part numbering is very confusing. 
     A8M415 is from the web page title.
     Alternate numbers include 
     "Model Number" (in datasheet): A8MS15
  - The main difference with the H8MF15 seems to be the 
    controller. Detailed differences are unknown.
#. [5x PTZ module][PTZ_module]
#. [Hangzhou Xiongmai IPG-HP500NS-A][XM_HP500]
   - XM seems to be a large OEM in China
#. [Camhi IMX335][Camhi_IMX335]
   - [Camhi seems to be an OEM][Camhi]
   - This module seems to be very common across Aliexpress
#. [SMTSEC SIP-E226K][SMTSEC_SIP_E226K]
   - [SMTSEC seems to be an OEM][SMTSEC]
   - Sony STARVIS
#. [Ansice bullet camera]
   - Another enclosed option with IR
#. [BlueFishCam 4MP enclosed camera][BlueFishCam_tilt]
   - Ansice is likely the OEM
#. [ELP-USB48MP01-AF70][ELP_Ali]
   - High resolution Samsung Sensor
   - USB
   - [ELP is an OEM][ELP]

Currently, the top choice is the Ansice B604 - it closely 
matches the requirements.

It is noted that modules from reputable brands which are 
[distributed by Digikey][Digikey_camera_modules] are 
quite costly for the same specifications.

With no official dimensions for the module, measurements 
on an image give 1.016/6*212 = 35.9mm (assumed 0402 measured
 at 6 pixels)
 
# Camera Module Documentation

Ultimately, the Ansice H8M415 was chosen with a 2.8mm lens 
purchased from cambase on Amazon.

The datasheet was not actually found before purchasing but 
[is available][Ansice_H8M415_ds] and lists pinout info.

Additionally, the RJ45 adapter module was disassembled to 
show that the POE circuitry is not on the camera module but 
in the RJ45 adapter module (TODO link image).

While the POE was purchased since it was available,
it is suspected that the non-POE option consists of 
different pin assembly on the connectors. Probing reveals 
that the DC power is shorted inside the cable assebly.

DC barrel outer  -> 2pin black pin
DC barrel center -> 2pin red pin
DC barrel outer  -> 8pin black pin
DC barrel center -> 8pin red pin

The exact connector series that is used across the module 
could not be found after an extensive search.
These connectors are (mostly):

- vertical
- 1.25mm pitch
  - Note than initial incorrect pitch calculation indicated 
    a 1mm pitch and a lengthy search was done for this pitch
    (and other features) with no results.
  - Measurements done on the 8 pin connector for lowest 
    errors. Other connectors seem to match.
- shrouded on only 3 sides
- have soldered mechanical supports on the sides

The closest that could be found was:
TE Connectivity 1734260-8 but this is 1.25mm pitch
Other compatible connectors include

- Molex Picoblade
- [Sunny Young R6501-xxTS-SMT][SY_R6500]
- Hirose DF13
- Amphenol ICC 10114828-10108LF
  - Specific datasheet correspondence done with measurements
    for this connector series (10114826-00008LF datasheet 
    with female measurements)

Note that the 2pin 12V DC connector has a wider pitch than 
all other connectors.

Per [the image on the website][STC_Picoblade], 
STC Cable seems to manufacture
 OEM cables with the correct exact connector series (per 
mechanical solder support style). This suggests that this 
particular model is only available in China or to OEMs and 
not from a major connector brand (JST, Hirose, etc.).

Camera power consumption while streaming fluctuates between 150 and 190mA.

# Camera Configuration

# Network Design

## Architecture

Simplest:

camera 1 - main network switch
camera 2 - main network switch
computer - main network switch
DHCP appliance (optional) - main network switch

Direct:

camera 1 - Ethernet-USB adapter - computer USB 1
camera 2 - Ethernet-USB adapter - computer USB 2

    Low latency
    We have 1x adapter already
    No direct remote access (viewing camera on computer over VNC possible though)

Dedicated Network:

camera 1 - local switch 1
camera 2 - local switch 1
local switch 1 - Ethernet-USB adapter - computer USB 1

    Scales to more cameras if needed
    Less network congestion due to dedicated camera network leads to fewer dropped frames (a dedicated network is usually recommended for cameras and phones)
    Additional switch needed (+45 CAD) or I can give you a spare WiFi router I have.
    Router can be used instead of switch or software DHCP server can be run

## Camera Placement Security

When used in public spaces (including many workplaces), the 
placement of the camera and what is visible may require not 
only IT approval but site security and HR approval to 
prevent the recording of people and/or sensitive information.

## Network Security

The IT department may need to grant access (static IP or 
MAC address) to the network before a device is assigned an 
IP address and/or allowed through a firewall.

IT may also be able to assign a static IP which would be 
preferred for a fixed camera installation.

These issues may be bypassed if a private network (isolated 
from IT managed network, own routing equipment) is 
deployed but this typically prevents internet access.

Another way to bypass this is to connect Ethernet cameras 
directly to a computer with a USB-Ethernet adapter.

The MAC address of the initial camera is: F6:70:00:0A:B7:09

## Camera Access Security

Be sure to change the default username and/or password of 
the camera administration page at final deployment to 
prevent spying. 

# Initial Camera Access

The camera most likely needs to be reconfigured for the network.

Since the camera is delivered with a static IP address, it 
may not be able to connect to an existing network.

To work around this, a dedicated router was reconfigured to 
enable access to the camera for reconfiguration. Specifically,
the router address was set to:

Router IP  : 192.168.1.1 (from 192.168.0.1)
Subnet Mask: 255.255.255.0

The camera static IP was 192.168.1.10 which was outside of 
the default subnet mask. An odd quirk in the router prevented 
switching the mask to 255.255.0.0 so the router IP (subnet) 
was changed instead.


# Circuit Design

## LEDs

The LEDs are white. Probing showed that the 12 LEDs they 
are in parallel groups of 3.
A 10V drive test voltage lit all LEDs brightly @ 50mA current draw.
2 LEDs were much high efficiency and lit with little voltage.
Damage was suspected until this was realized.

This is inline with expectations:
[An InGaN White LED has V_f = 3.1 V][QTB_QBLP601]
3.1*3 = 9.3 V

## Pinouts

Note that the harness that is provided with the camera has 
no wire for the network indicator light (as indicated in 
the camera datasheet).

To reuse the harness, this signal is omitted.

# Manufacturing Instructions

Follow the colour codes and wire types indicated in the 
following table:

------------------- ----------------------- ------------
12V_P               EXISTING CAMERA HARNESS RED
GND                 EXISTING CAMERA HARNESS BLACK
ETHERNET_TX_P       EXISTING CAMERA HARNESS WHITE-BLUE
ETHERNET_TX_N       EXISTING CAMERA HARNESS BLUE
ETHERNET_RX_P       EXISTING CAMERA HARNESS WHITE-GREEN
ETHERNET_RX_N       EXISTING CAMERA HARNESS GREEN
RESET               EXISTING CAMERA HARNESS PURPLE
12V_OUT_P           22 or 24 AWG STRANDED   RED
GND_OUT             22 or 24 AWG STRANDED   BLACK
ETHERNET_TX_OUT_P   CAT5e or CAT6           WHITE-GREEN
ETHERNET_TX_OUT_N   CAT5e or CAT6           GREEN
ETHERNET_RX_OUT_P   CAT5e or CAT6           WHITE-ORANGE
ETHERNET_RX_OUT_N   CAT5e or CAT6           ORANGE
RESET_OUT           24 AWG stranded         WHITE
LED_P               24 AWG stranded         RED
LED_N               24 AWG stranded         BLACK
LED_OUT_P           24 AWG stranded         RED
LED_OUT_N           24 AWG stranded         BLACK
TEMP_SENSE0         24 AWG stranded         WHITE
TEMP_SENSE1         24 AWG stranded         WHITE
TEMP_SENSE_OUT0     24 AWG stranded         WHITE
TEMP_SENSE_OUT1     24 AWG stranded         WHITE
HEATER0             24 AWG stranded         WHITE
HEATER1             24 AWG stranded         WHITE
HEATER_OUT0         24 AWG stranded         WHITE
HEATER_OUT1         24 AWG stranded         WHITE
------------------- ----------------------- ------------

- Ethernet colouring follows EIA T568A
- White wire colour may be substituded for any other colour 
  across the entire assembly.
- Twist Ethernet wires within chassis
- Remove 2 pin power and foild shield from existing camera 
  harness.
- Lengths:
  Existing Camera Harness  70mm  slack
  CAT5e or CAT6            100mm slack
- Use Pb Free solder only
- After assembly and testing, label assemblies with
  TRACKING ID x (HARNESS x SN x)
  TRACKING ID x (ASSY x SN x)
  
  
# Release Notes

## 20230831102000

- Initial release

Ali_IP_cam1: https://www.aliexpress.com/item/1005005319648835.html
Ali_IP_cam2: https://www.aliexpress.com/item/1005005459614590.html
Ali_IP_cam3: https://www.aliexpress.com/item/4000762000490.html
REVO_I704_P: https://www.amazon.ca/REVODATA-UltraHD-Security-Waterproof-Detection/dp/B0B17FHLV6
BlueFishCam_5MP: https://www.amazon.ca/Camera-Network-ModuleBlueFishcam-Security-Upgraded/dp/B088BQJFWD
Ansice_H8M415_Amazon: https://www.amazon.ca/Camera-Network-Megapixel-IR-Cut-Professional/dp/B09X1H3653
Ansice_H8M415: https://www.ansice.net/en-ca/products/super-wide-view-fisheye-lens-4k-ipc-camera-poe-sony-imx415-8m-sensor-ip-camera-onvif-ip-security-cctv-camera-aj4k415
Ansice_A8M415 : https://www.ansice.net/en-ca/collections/ip-cameras/products/4k-ipc-pcb-board-camera-poe-sony-imx415-8m-sensor-ip-camera-main-board-onvif-ip-security-cctv-board-camera-for-spy-diy-upgrade#description
Ansice_M4MN41: https://www.ansice.net/en-ca/collections/ip-cameras/products/1-0mp-720p-wired-ip-camera-lens-3-6mm-onvif-ip-security-cctv-board-camera-for-professional-diy?variant=31384309104693
Ansice_M8MN82_Amazon: https://www.amazon.ca/Camera-Ansice-Security-Surveillance-Network/dp/B09X1RGW7G
Ansice_M8MN82: https://www.ansice.net/en-ca/collections/ip-cameras/products/8mp-4k-pcb-board-camera-ip-camera-main-board-for-professional-diy
PTZ_module: https://www.amazon.ca/Starlight-Wireless-Network-2-7-13-5mm-Upgrade/dp/B0C2CN76FV
REVODATA: https://www.revoamerica.com/
XM_HP500: https://www.xiongmaitech.com/en/index.php/product/product-detail/3/101/335
Camhi: https://camhi.pro/camera/
Camhi_IMX335: https://www.aliexpress.com/item/4000921484912.html
SMTSEC_SIP_E226K: http://www.smtsec.com/productinfo/654691.html
SMTSEC: http://www.smtsec.com/home
Ansice_bullet: https://www.amazon.ca/Camera-Network-Ansice-Security-Weatherproof/dp/B08931PZK3
BlueFishCam_tilt: https://www.amazon.ca/BluefishCam-Camera-Waterproof-Varifocal-Network/dp/B08NYJ2FH7
Digikey_camera_modules: https://www.digikey.ca/en/products/filter/camera-modules/1003
ELP_Ali: https://www.aliexpress.com/item/1005005384311446.html
ELP: http://www.elpcctv.com/
Ansice_H8M415_ds: https://cdn.shopifycdn.net/s/files/1/0031/0994/5408/files/H8MF15.pdf?v=1670405221
SY_R6500 : https://www.sunnyyoung.com.tw/en/product-626919/1-25mm-Pitch-Wire-to-Board-Connector-R6500-Series.html
STC_Picoblade : https://www.stc-cable.com/pitch-1-25mm-molex-51021-51047-wire-to-board-connector-and-cable.html
QTB_QBLP_601 : https://www.qt-brightek.com/datasheet/QBLP601_series.pdf
