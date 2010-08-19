
How to install the needed dependencies :
----------------------------------------

ArToolKitPlus 2.1.1 :
---------------------

- download http://studierstube.icg.tu-graz.ac.at/handheld_ar/download/ARToolKitPlus_2.1.1.zip
- unzip in this folder
- resulting structure : $(SolutionDir)\Dependencies\ARToolKitPlus_2.1.1\include\ARToolKitPlus\ARToolKitPlus.h

ofVideoInput :
--------------

- download openFrameworks FAT visual 2008 : http://www.openframeworks.cc/download
- extract third party plugins : ofVideoInput in this folder
- resulting structure : 
			- $(SolutionDir)\Dependencies\ofVideoInput\include\videoInput.h
			- $(SolutionDir)\Dependencies\ofVideoInput\lib\videoInput.lib

DevIL :
-------
- download "DevIL 1.7.8 SDK for 32-bit Windows" from http://openil.sourceforge.net/download.php
- unzip in $(SolutionDir)\Dependencies\DevIL\
- resulting structure :
                       - $(SolutionDir)\Dependencies\DevIL\lib\DevIL.dll

cudpp v1.1.1 :
--------------
- download "cudpp_src_1.1.1.zip" from http://code.google.com/p/cudpp/downloads/list
- unzip in $(SolutionDir)\Dependencies\
- resulting structure : (you need to rename "cudpp_src_1.1.1" in "cudpp_1.1.1")
                       - $(SolutionDir)\Dependencies\cudpp_1.1.1\include\cudpp.h
                       - $(SolutionDir)\Dependencies\cudpp_1.1.1\lib\cudpp32.dll

jpeg:
-----
- I have taken the version included with Bundler 0.4 (http://phototour.cs.washington.edu/bundler/)
- resulting structure :
					   - $(SolutionDir)\Dependencies\jpeg\src\jversion.h		

JSON Spirit : 
-------------
- download "json_spirit_v4.03.zip" from http://www.codeproject.com/KB/recipes/JSON_Spirit.aspx
- unzip in $(SolutionDir)\Dependencies\
- resulting structure : (you need to rename "json_spirit_v4.03" in "json_spirit")
					   - $(SolutionDir)\Dependencies\json_spirit\json.sln