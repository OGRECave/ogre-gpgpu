# General Purpose GPU Computing with Ogre

The purpose of this libray is to abstract DirectX & OpenGL interop with Cuda and OpenCL. It means that you can use `Ogre::Texture` and `Ogre::HardwareVertexBuffer` without having to bother which `Ogre::RenderSystem` is active (DX9, DX10 or GL).

[![Demo
video](https://img.youtube.com/vi/0KkB38CB3vY/0.jpg)](https://www.youtube.com/watch?v=0KkB38CB3vY)

There are "Property Sheets" (.vsprops) in almost all project.
So you need to adapt theirs "User Macros" path to your needs.

Before running any project you need to launch "prepare bin folder" (needs OGRE_HOME to be defined in the Ogre.vsprops)
It will copy dll from ogre folder to the corresponding binary folder.

