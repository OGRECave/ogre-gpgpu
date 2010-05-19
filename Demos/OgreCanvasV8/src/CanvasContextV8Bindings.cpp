#include "CanvasContextV8Bindings.h"

#include "V8CanvasFunctions.h"

using namespace v8;

/**************************
 *  		Helper		  *
 **************************/

OgreCanvas::CanvasContext*            CanvasContextV8Bindings::context2D = NULL;
v8::Persistent<v8::Context>           CanvasContextV8Bindings::contextV8;
v8::Persistent<v8::FunctionTemplate>  CanvasContextV8Bindings::canvasGradientTemplate;
v8::Persistent<v8::FunctionTemplate>  CanvasContextV8Bindings::canvasPatternTemplate;
v8::Persistent<v8::FunctionTemplate>  CanvasContextV8Bindings::imageTemplate;
v8::Persistent<v8::FunctionTemplate>  CanvasContextV8Bindings::consoleTemplate;

CanvasContextV8Bindings::CanvasContextV8Bindings()
{}

void CanvasContextV8Bindings::loadScript(const std::string& _filename, OgreCanvas::CanvasContext* _canvasContext, OgreCanvas::CanvasLogger* _console)
{
	CanvasContextV8Bindings::context2D = _canvasContext;
	
	HandleScope handle_scope;
	
	//Console :
		
		//template
		Handle<FunctionTemplate> consoleTemplate = FunctionTemplate::New();
		consoleTemplate->SetClassName(v8::String::New("Console"));
		CanvasContextV8Bindings::consoleTemplate = Persistent<FunctionTemplate>::New(consoleTemplate);

		//prototype
		Handle<ObjectTemplate> consolePrototype = consoleTemplate->PrototypeTemplate();

		//attaching method
		consolePrototype->Set("log", FunctionTemplate::New(log));

		//creating instance
		Handle<ObjectTemplate> consoleInstance = consoleTemplate->InstanceTemplate();
		consoleInstance->SetInternalFieldCount(1);

	//Image :
		
		//template
		Handle<FunctionTemplate> imageTemplate = FunctionTemplate::New();
		imageTemplate->SetClassName(v8::String::New("Image"));
		CanvasContextV8Bindings::imageTemplate = Persistent<FunctionTemplate>::New(imageTemplate);

		//prototype
		Handle<ObjectTemplate> imagePrototype = imageTemplate->PrototypeTemplate();

		//creating instance
		Handle<ObjectTemplate> imageInstance = imageTemplate->InstanceTemplate();
		imageInstance->SetInternalFieldCount(1);

	//Canvas gradient :
		
		//template
		Handle<FunctionTemplate> canvasGradientTemplate = FunctionTemplate::New();
		canvasGradientTemplate->SetClassName(v8::String::New("CanvasGradient"));
		CanvasContextV8Bindings::canvasGradientTemplate = Persistent<FunctionTemplate>::New(canvasGradientTemplate);

		//prototype
		Handle<ObjectTemplate> canvasGradientPrototype = canvasGradientTemplate->PrototypeTemplate();

		//creating instance
		Handle<ObjectTemplate> canvasGradientInstance = canvasGradientTemplate->InstanceTemplate();
		canvasGradientInstance->SetInternalFieldCount(1);

		//attaching method
		canvasGradientPrototype->Set("addColorStop", FunctionTemplate::New(addColorStop));

	//Canvas Pattern :

		//template
		Handle<FunctionTemplate> canvasPatternTemplate = FunctionTemplate::New();
		canvasPatternTemplate->SetClassName(v8::String::New("CanvasPattern"));
		CanvasContextV8Bindings::canvasPatternTemplate = Persistent<FunctionTemplate>::New(canvasPatternTemplate);

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
			canvasContextPrototype->Set("saveToPNG",   FunctionTemplate::New(saveToPNG));		
			canvasContextPrototype->Set("clear",       FunctionTemplate::New(clear));

		//creating instance
		Handle<ObjectTemplate> canvasContextInstance = canvasContextTemplate->InstanceTemplate();
		canvasContextInstance->SetInternalFieldCount(1);

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
	global->Set(String::New("loadImage"), FunctionTemplate::New(loadImage));

	Persistent<Context> context = Context::New(NULL, global);
	CanvasContextV8Bindings::contextV8 = context;

	Context::Scope context_scope(context);

		//building link between js 'ctx' variable and c++ _canvasContext variable
		Handle<Function> canvasContextConstructor = canvasContextTemplate->GetFunction();
		Local<Object> obj = canvasContextConstructor->NewInstance();
		obj->SetInternalField(0, External::New(_canvasContext));
		context->Global()->Set(v8::String::New("ctx"), obj);

		//building link between js 'console' variable and c++ _console variable
		Handle<Function> consoleConstructor = consoleTemplate->GetFunction();
		Local<Object> obj2 = consoleConstructor->NewInstance();
		obj2->SetInternalField(0, External::New(_console));
		context->Global()->Set(v8::String::New("console"), obj2);

	Handle<v8::String> source = v8::String::New(readScript(_filename).c_str());
	Handle<Script> script = Script::Compile(source);
	Handle<Value> result = script->Run();
	/*
	CanvasContextV8Bindings::canvasGradientTemplate.Dispose();
	CanvasContextV8Bindings::canvasPatternTemplate.Dispose();
	CanvasContextV8Bindings::imageTemplate.Dispose();
	*/
}

void CanvasContextV8Bindings::executeJS(const std::string& _js)
{
	HandleScope handle_scope;
	Context::Scope context_scope(CanvasContextV8Bindings::contextV8);

	Handle<v8::String> source = v8::String::New(_js.c_str());
	Handle<Script> script = Script::Compile(source);
	Handle<Value> result = script->Run();
}

void CanvasContextV8Bindings::dispose()
{
	CanvasContextV8Bindings::contextV8.Dispose();
}

std::string CanvasContextV8Bindings::readScript(const std::string& _filename)
{
	std::stringstream data;
	std::ifstream input(_filename.c_str());
	while(!input.eof())
	{
		std::string line;
		std::getline(input, line);
		data << line <<std::endl;
	}
	return data.str();
}