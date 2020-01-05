/*!

\mainpage <A href="http://www.visual-experiments.com/demos/ogrecuda/">Ogre::Cuda</A>
\section intro Presentation

<IMG SRC="http://www.visual-experiments.com/blog/wp-content/uploads/2010/03/nvidia-cuda.jpg" ALT="Cuda" />
The purpose of this libray is to abstract DirectX & OpenGL interop with Cuda.<BR>
It means that you can use Ogre::Texture and Ogre::HardwareVertexBuffer <BR>
with Cuda without having to bother which Ogre::RenderSystem is active (DX9, DX10 or GL).<BR>

\section utilization Utilization scheme : 
\subsection simple Simple example with one texture to map
<PRE>
Ogre::TexturePtr tex;
Ogre::Cuda::Root* mCudaRoot = Ogre::Cuda::Root::createRoot(renderWindow, renderSystem);
mCudaRoot->init();

Ogre::Cuda::Texture* mCudaTex = mCudaRoot->getTextureManager()->createTexture(tex);

//init
mCudaTex->registerForCudaUse();

//on each Cuda update
mCudaTex->map();
Ogre::Cuda::TextureDeviceHandle textureHandle = mCudaTex->getDeviceHandle(0, 0);
mCudaTex->updateReading(textureHandle);
cudaFunction(textureHandle->getPointer());
mCudaTex->updateWriting(textureHandle);
mCudaTex->unmap();

//shutdown
mCudaTex->unregister();
mCudaRoot->getTextureManager()->destroyTexture(mCudaTex);
mCudaRoot->shutdown();
</PRE>

\subsection multiple Efficient way to map multiples CudaRessource (texture, vertex buffer) in one call
<PRE>
std::vector<Ogre::TexturePtr> textures;
std::vector<Ogre::Cuda::CudaRessource*> ressources;
Ogre::Cuda::Root* mCudaRoot = Ogre::Cuda::Root::createRoot(renderWindow, renderSystem);
mCudaRoot->init();

for (unsigned int i=0; i<textures.size(); ++i)
	ressources.push_back(mCudaRoot->getTextureManager()->createTexture(textures[i]);

//init
for (unsigned int i=0; i<ressources.size(); ++i)
	ressources[i]->registerForCudaUse();

//on each Cuda update
mCudaRoot->map(ressources); //efficient way to map multiple ressources in one call	
for (unsigned int i=0; i<ressources.size(); ++i)
{
	if (ressources[i]->getType() == Ogre::Cuda::TEXTURE_RESSOURCE)
	{
		Ogre::Cuda::TextureDeviceHandle textureHandle = static_cast<Ogre::Cuda::Texture*>(ressources[i])->getDeviceHandle(0, 0);
		ressources[i]->updateReading(textureHandle);
		cudaTextureFunction(textureHandle->getPointer());
		ressources[i]->updateWriting(textureHandle);
	}
	else
		cudaVertexBufferFunction(static_cast<Ogre::Cuda::VertexBuffer*>(ressources[i])->getPointer());
}	
mCudaRoot->unmap(ressources);

//shutdown
for (unsigned int i=0; i<ressources.size(); ++i)
{
	ressources[i]->unregister();
	if (ressources[i]->getType() == Ogre::Cuda::TEXTURE_RESSOURCE)
	{
		mCudaRoot->getTextureManager()->destroyTexture(ressources[i]);	
	}
	else
		mCudaRoot->getVertexBufferManager ()->destroyVertexBuffer(ressources[i]);	
}
mCudaRoot->shutdown();
</PRE>
*/