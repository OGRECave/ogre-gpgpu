/*
	Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32

#include "OgreGPGPUD3D9.h"

using namespace Ogre::GPGPU;

D3D9Root::D3D9Root(Ogre::RenderWindow* renderWindow)
: Root()
{
	renderWindow->getCustomAttribute("D3DDEVICE", (void*) &mDevice);
}

void D3D9Root::waitForCompletion()
{
	//MSDN Queries (Direct3D 9) : http://msdn.microsoft.com/en-us/library/bb147308(VS.85).aspx
	//GPGPU Forum : http://www.ibiblio.org/harrism/phpBB2/viewtopic.php?t=4032&view=previous&sid=bc32eb693cdbfaef1ac0cc62d3444af2
	IDirect3DQuery9* query;
	mDevice->CreateQuery(D3DQUERYTYPE_EVENT, &query); 

	//Flush command buffer 
	query->Issue(D3DISSUE_END); 
	query->GetData(NULL, 0, D3DGETDATA_FLUSH); 

	while (S_FALSE == query->GetData(NULL, 0, D3DGETDATA_FLUSH))
		Sleep(0);
}

#endif //if OGRE_PLATFORM == OGRE_PLATFORM_WIN32