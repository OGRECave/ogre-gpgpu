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

#include "CanvasV8Context.h"

#include "CanvasV8Functions.h"

using namespace v8;

Ogre::Canvas::V8Context::V8Context(Canvas::Context* context, Canvas::Logger* console)
{
	mContext = context;
	mLogger  = console;

	bindCanvasContext();
}

Ogre::Canvas::V8Context::~V8Context()
{	
	mCanvasGradientTemplate.Dispose();
	mCanvasGradientTemplate.Clear();

	mCanvasPatternTemplate.Dispose();
	mCanvasPatternTemplate.Clear();

	mImageTemplate.Dispose();
	mImageTemplate.Clear();

	mConsoleTemplate.Dispose();
	mConsoleTemplate.Clear();

	mV8Context.Dispose();
	mV8Context.Clear();
}

void Ogre::Canvas::V8Context::execute(const std::string& javascript)
{
	HandleScope handle_scope;
	v8::Context::Scope context_scope(mV8Context);

	Handle<v8::String> source = v8::String::New(javascript.c_str());
	Handle<Script> script = Script::Compile(source);
	Handle<Value> result = script->Run();
}

std::string Ogre::Canvas::V8Context::readScript(const std::string& filename)
{
	std::stringstream data;
	std::ifstream input(filename.c_str());
	while(!input.eof())
	{
		std::string line;
		std::getline(input, line);
		data << line <<std::endl;
	}
	return data.str();
}

void Ogre::Canvas::V8Context::bindCanvasContext()
{
	HandleScope handle_scope;

	if (mLogger)	
	{
	
	//Console :
		
		//template
		Handle<FunctionTemplate> consoleTemplate = FunctionTemplate::New();
		consoleTemplate->SetClassName(v8::String::New("Console"));
		mConsoleTemplate = Persistent<FunctionTemplate>::New(consoleTemplate);

		//prototype
		Handle<ObjectTemplate> consolePrototype = consoleTemplate->PrototypeTemplate();

		//attaching method
		consolePrototype->Set("log", FunctionTemplate::New(log));

		//creating instance
		Handle<ObjectTemplate> consoleInstance = consoleTemplate->InstanceTemplate();
		consoleInstance->SetInternalFieldCount(1);
	}

	//Image :
		
		//template
		Handle<FunctionTemplate> imageTemplate = FunctionTemplate::New();
		imageTemplate->SetClassName(v8::String::New("Image"));
		mImageTemplate = Persistent<FunctionTemplate>::New(imageTemplate);

		//prototype
		Handle<ObjectTemplate> imagePrototype = imageTemplate->PrototypeTemplate();

		//creating instance
		Handle<ObjectTemplate> imageInstance = imageTemplate->InstanceTemplate();
		imageInstance->SetInternalFieldCount(1);

	//Canvas gradient :
		
		//template
		Handle<FunctionTemplate> canvasGradientTemplate = FunctionTemplate::New();
		canvasGradientTemplate->SetClassName(v8::String::New("CanvasGradient"));
		mCanvasGradientTemplate = Persistent<FunctionTemplate>::New(canvasGradientTemplate);

		//prototype
		Handle<ObjectTemplate> canvasGradientPrototype = canvasGradientTemplate->PrototypeTemplate();

		//creating instance
		Handle<ObjectTemplate> canvasGradientInstance = canvasGradientTemplate->InstanceTemplate();
		canvasGradientInstance->SetInternalFieldCount(1);

		//attaching method
		canvasGradientPrototype->Set("addColorStop", FunctionTemplate::New(addColorStop));
		canvasGradientPrototype->Set("destroy",      FunctionTemplate::New(gradient_destroy));

	//Canvas Pattern :

		//template
		Handle<FunctionTemplate> canvasPatternTemplate = FunctionTemplate::New();
		canvasPatternTemplate->SetClassName(v8::String::New("CanvasPattern"));
		mCanvasPatternTemplate = Persistent<FunctionTemplate>::New(canvasPatternTemplate);

		//prototype
		Handle<ObjectTemplate> canvasPatternPrototype = canvasPatternTemplate->PrototypeTemplate();
		
		//creating instance
		Handle<ObjectTemplate> canvasPatternInstance = canvasPatternTemplate->InstanceTemplate();
		canvasPatternInstance->SetInternalFieldCount(1);

	//Canvas context :

		//template
		Handle<FunctionTemplate> canvasContextTemplate = FunctionTemplate::New();
		canvasContextTemplate->SetClassName(v8::String::New("CanvasContext"));
	
		//prototype
		Handle<ObjectTemplate> canvasContextPrototype = canvasContextTemplate->PrototypeTemplate();

		//attaching method

			//2D Context
			canvasContextPrototype->Set("save",        FunctionTemplate::New(save));
			canvasContextPrototype->Set("restore",     FunctionTemplate::New(restore));

			//Transformation
			canvasContextPrototype->Set("scale",        FunctionTemplate::New(scale));
			canvasContextPrototype->Set("rotate",       FunctionTemplate::New(rotate));
			canvasContextPrototype->Set("translate",    FunctionTemplate::New(translate));
			canvasContextPrototype->Set("transform",    FunctionTemplate::New(transform));
			canvasContextPrototype->Set("setTransform", FunctionTemplate::New(setTransform));
			
			//Image drawing
			canvasContextPrototype->Set("drawImage",    FunctionTemplate::New(drawImage));		

			//Colors, styles and shadows
			canvasContextPrototype->Set("createLinearGradient", FunctionTemplate::New(createLinearGradient));
			canvasContextPrototype->Set("createRadialGradient", FunctionTemplate::New(createRadialGradient));
			canvasContextPrototype->Set("createPattern",        FunctionTemplate::New(createPattern));

			//Paths
			canvasContextPrototype->Set("beginPath",        FunctionTemplate::New(beginPath));
			canvasContextPrototype->Set("closePath",        FunctionTemplate::New(closePath));		
			canvasContextPrototype->Set("fill",             FunctionTemplate::New(fill));
			canvasContextPrototype->Set("stroke",           FunctionTemplate::New(stroke));
			canvasContextPrototype->Set("clip",             FunctionTemplate::New(clip));

			canvasContextPrototype->Set("moveTo",           FunctionTemplate::New(moveTo));
			canvasContextPrototype->Set("lineTo",           FunctionTemplate::New(lineTo));
			canvasContextPrototype->Set("quadraticCurveTo", FunctionTemplate::New(quadraticCurveTo));
			canvasContextPrototype->Set("bezierCurveTo",    FunctionTemplate::New(bezierCurveTo));
			canvasContextPrototype->Set("arcTo",            FunctionTemplate::New(arcTo));
			canvasContextPrototype->Set("arc",              FunctionTemplate::New(arc));
			canvasContextPrototype->Set("rect",             FunctionTemplate::New(rect));
			canvasContextPrototype->Set("isPointInPath",    FunctionTemplate::New(isPointInPath));

			//Text
			canvasContextPrototype->Set("fillText",    FunctionTemplate::New(fillText));
			canvasContextPrototype->Set("strokeText",  FunctionTemplate::New(strokeText));
			canvasContextPrototype->Set("measureText", FunctionTemplate::New(measureText));

			//Rectangles
			canvasContextPrototype->Set("clearRect",   FunctionTemplate::New(clearRect));
			canvasContextPrototype->Set("fillRect",    FunctionTemplate::New(fillRect));
			canvasContextPrototype->Set("strokeRect",  FunctionTemplate::New(strokeRect));

			//New
			canvasContextPrototype->Set("saveToFile",  FunctionTemplate::New(saveToFile));		
			canvasContextPrototype->Set("loadImage",   FunctionTemplate::New(loadImage));


		//creating instance
		Handle<ObjectTemplate> canvasContextInstance = canvasContextTemplate->InstanceTemplate();
		canvasContextInstance->SetInternalFieldCount(2);

		//attaching properties

			//Compositing
			canvasContextInstance->SetAccessor(v8::String::New("globalAlpha"), getterGlobalAlpha, setterGlobalAlpha);
			canvasContextInstance->SetAccessor(v8::String::New("globalCompositeOperation"), getterGlobalCompositeOperation, setterGlobalCompositeOperation);
			
			//Line styles
			canvasContextInstance->SetAccessor(v8::String::New("lineWidth"),   getterLineWidth,  setterLineWidth);
			canvasContextInstance->SetAccessor(v8::String::New("lineCap"),     getterLineCap,    setterLineCap);
			canvasContextInstance->SetAccessor(v8::String::New("lineJoin"),    getterLineJoin,   setterLineJoin);
			canvasContextInstance->SetAccessor(v8::String::New("miterLimit"),  getterMiterLimit, setterMiterLimit);
			canvasContextInstance->SetAccessor(v8::String::New("lineDash"),    getterLineDash,   setterLineDash);
			
			//Colors, styles and shadows
			canvasContextInstance->SetAccessor(v8::String::New("fillStyle"),     getterFillStyle,     setterFillStyle);
			canvasContextInstance->SetAccessor(v8::String::New("strokeStyle"),   getterStrokeStyle,   setterStrokeStyle);
			canvasContextInstance->SetAccessor(v8::String::New("shadowOffsetX"), getterShadowOffsetX, setterShadowOffsetX);
			canvasContextInstance->SetAccessor(v8::String::New("shadowOffsetY"), getterShadowOffsetY, setterShadowOffsetY);
			canvasContextInstance->SetAccessor(v8::String::New("shadowBlur"),    getterShadowBlur,    setterShadowBlur);
			canvasContextInstance->SetAccessor(v8::String::New("shadowColor"),   getterShadowColor,   setterShadowColor);

			//Text
			canvasContextInstance->SetAccessor(v8::String::New("font"),         getterFont,         setterFont);
			canvasContextInstance->SetAccessor(v8::String::New("textAlign"),    getterTextAlign,    setterTextAlign);
			canvasContextInstance->SetAccessor(v8::String::New("textBaseline"), getterTextBaseline, setterTextBaseline);
			
			//New
			canvasContextInstance->SetAccessor(v8::String::New("antialiasing"), getterAntiAliasing, setterAntiAliasing);

	//Image
	Handle<ObjectTemplate> global = ObjectTemplate::New();
	//global->Set(String::New("loadImage"), FunctionTemplate::New(loadImage));

	Persistent<v8::Context> context = v8::Context::New(NULL, global);
	mV8Context = context;

	v8::Context::Scope context_scope(context);

		//building link between js 'ctx' variable and c++ mContext
		Handle<Function> canvasContextConstructor = canvasContextTemplate->GetFunction();
		Local<Object> obj = canvasContextConstructor->NewInstance();
		obj->SetPointerInInternalField(0, mContext);
		obj->SetPointerInInternalField(1, this);
		context->Global()->Set(v8::String::New("ctx"), obj);

	if (mLogger)
	{
		//building link between js 'console' variable and c++ mLogger
		Handle<Function> consoleConstructor = mConsoleTemplate->GetFunction();
		Local<Object> obj = consoleConstructor->NewInstance();
		obj->SetPointerInInternalField(0, mLogger);
		context->Global()->Set(v8::String::New("console"), obj);
	}
}