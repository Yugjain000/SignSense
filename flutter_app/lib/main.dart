import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:permission_handler/permission_handler.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (_) {
    // If cameras cannot be queried on startup (common on Web before user grants permission),
    // DashboardScreen._initializeCameraWithPermissions will safely initialize it.
  }
  runApp(SignLanguageApp());
}

class SignLanguageApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ISL Assistant Pro',
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: Color(0xFF121212),
        primaryColor: Color(0xFF6C63FF),
        colorScheme: ColorScheme.dark(
          primary: Color(0xFF6C63FF),
          secondary: Color(0xFF03DAC6),
          surface: Color(0xFF1E1E1E),
        ),
        fontFamily: 'Inter',
      ),
      home: DashboardScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DashboardScreen extends StatefulWidget {
  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> with SingleTickerProviderStateMixin {
  CameraController? controller;
  FlutterTts flutterTts = FlutterTts();
  
  List<String> currentWords = [];
  String displaySentence = "Awaiting sign language input...";
  bool isRecording = false;
  bool autoSpeakEnabled = false;

  // THEME VARIABLES
  bool isDarkMode = true;
  Color get bgColor => isDarkMode ? Color(0xFF050511) : Color(0xFFF5F5F7);
  Color get panelColor => isDarkMode ? Color(0xFF101018) : Colors.white;
  Color get textColor => isDarkMode ? Colors.white : Colors.black87;
  Color get subtitleColor => isDarkMode ? Colors.white70 : Colors.black54;
  Color get borderColor => isDarkMode ? Colors.white10 : Colors.black12;
  Color get inputBgColor => isDarkMode ? Color(0xFF0A0A12) : Color(0xFFF0F0F5);
  Color get accentColor => Color(0xFF4A62FF);
  
  stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  String _hearingPersonResponse = "Tap microphone to dictate...";
  
  String currentDetection = "Camera Off / Inactive";
  List<List<Offset>> currentLandmarks = [];
  bool isMirrored = true;
  bool _isCapturing = false;

  Timer? frameTimer;
  DateTime? lastWordTime;
  Timer? silenceTimer;
  late AnimationController pulseController;

  @override
  void initState() {
    super.initState();
    pulseController = AnimationController(vsync: this, duration: Duration(milliseconds: 800))..repeat(reverse: true);
    
    _speech.initialize(onStatus: (status) {
      if (status == 'done' || status == 'notListening') {
        setState(() => _isListening = false);
      }
    });
    
    _initializeCameraWithPermissions();
  }

  Future<void> _initializeCameraWithPermissions() async {
    bool hasPermission = false;
    if (kIsWeb) {
      hasPermission = true;
    } else {
      Map<Permission, PermissionStatus> statuses = await [
        Permission.camera,
        Permission.microphone,
      ].request();
      hasPermission = statuses[Permission.camera] == PermissionStatus.granted;
    }

    if (hasPermission) {
      try {
        if (cameras.isEmpty) {
          cameras = await availableCameras();
        }
        if (cameras.isNotEmpty) {
          final frontCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
            orElse: () => cameras.first,
          );
          controller = CameraController(frontCamera, ResolutionPreset.low);
          await controller!.initialize();
          if (!mounted) return;
          setState(() {});
        }
      } catch (e) {
        print("Camera initialization error: $e");
      }
    }
  }

  @override
  void dispose() {
    pulseController.dispose();
    frameTimer?.cancel();
    silenceTimer?.cancel();
    controller?.dispose();
    super.dispose();
  }

  void _listenToSpeech() async {
    if (!_isListening) {
      bool available = await _speech.initialize();
      if (available) {
        setState(() => _isListening = true);
        _speech.listen(
          onResult: (val) => setState(() {
            _hearingPersonResponse = val.recognizedWords;
          }),
        );
      }
    } else {
      setState(() => _isListening = false);
      _speech.stop();
    }
  }

  void toggleRecording() {
    setState(() {
      isRecording = !isRecording;
      if (isRecording) {
        currentDetection = "Awaiting hand...";
        startFrameCapture();
      } else {
        frameTimer?.cancel();
        silenceTimer?.cancel();
        _isCapturing = false;
        currentDetection = "Camera Off / Inactive";
        currentLandmarks = [];
      }
    });
  }

  void startFrameCapture() {
    frameTimer = Timer.periodic(Duration(milliseconds: 150), (timer) async {
      if (!isRecording || controller == null || !controller!.value.isInitialized) return;
      if (_isCapturing) return;
      _isCapturing = true;
      try {
        final image = await controller!.takePicture();
        final bytes = await image.readAsBytes();
        final base64Image = base64Encode(bytes);
        
        final response = await http.post(
          Uri.parse('http://127.0.0.1:8000/process_frame'),
          headers: {"Content-Type": "application/json"},
          body: jsonEncode({"image_base64": base64Image}),
        );
        
        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);

          List<List<Offset>> parsedLandmarks = [];
          if (data['landmarks'] != null) {
            for (var hand in data['landmarks']) {
              List<Offset> pts = [];
              for (var pt in hand) {
                pts.add(Offset((pt[0] as num).toDouble(), (pt[1] as num).toDouble()));
              }
              parsedLandmarks.add(pts);
            }
          }

          bool hasHands = data['has_hands'] == true;
          String detectionText;
          if (!hasHands) {
            detectionText = "No hand detected";
          } else if (data['predicted_word'] != null) {
            detectionText = data['predicted_word'];
          } else if (data['raw_prediction'] != null) {
            detectionText = data['raw_prediction'];
          } else {
            detectionText = "Tracking hand...";
          }

          if (mounted) {
            setState(() {
              currentLandmarks = parsedLandmarks;
              currentDetection = detectionText;
            });
          }

          if (data['predicted_word'] != null) {
            if (lastWordTime == null || DateTime.now().difference(lastWordTime!).inSeconds >= 2) {
              if (mounted) {
                setState(() {
                  currentWords.add(data['predicted_word']);
                  lastWordTime = DateTime.now();
                  resetSilenceTimer();
                });
              }
            }
          }
        }
      } catch (e) {
        print("Frame error: $e");
      } finally {
        _isCapturing = false;
      }
    });
  }

  void resetSilenceTimer() {
    silenceTimer?.cancel();
    if (autoSpeakEnabled && currentWords.isNotEmpty) {
      silenceTimer = Timer(Duration(seconds: 3), () {
        generateSentence();
      });
    }
  }

  void removeWord(int index) {
    setState(() {
      currentWords.removeAt(index);
      resetSilenceTimer();
    });
  }

  void clearSentence() {
    setState(() {
      currentWords.clear();
      displaySentence = "Awaiting sign language input...";
      _hearingPersonResponse = "Tap microphone to dictate...";
      silenceTimer?.cancel();
    });
  }

  Future<void> generateSentence() async {
    if (currentWords.isEmpty) return;
    
    setState(() {
      displaySentence = "Translating semantics...";
    });
    
    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:8000/generate_sentence'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"words": currentWords}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          displaySentence = data['natural_sentence'];
        });
        await flutterTts.speak(displaySentence);
      }
    } catch (e) {
      setState(() => displaySentence = "Error connecting to AI Server.");
    }
  }



  Widget buildNavbar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 10.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text("SignSense", style: TextStyle(color: accentColor, fontSize: 24, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
          Row(
            children: [
              TextButton(onPressed: () {}, child: Text("Home", style: TextStyle(color: subtitleColor))),
              SizedBox(width: 8),
              Container(
                decoration: BoxDecoration(color: accentColor.withOpacity(0.2), borderRadius: BorderRadius.circular(8)),
                child: TextButton(onPressed: () {}, child: Text("Translate", style: TextStyle(color: accentColor))),
              ),
              SizedBox(width: 8),
              TextButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      backgroundColor: panelColor,
                      title: Text("About SignSense", style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
                      content: Text("SignSense is an AI-powered Two-Way Indian Sign Language Translator designed for our Final Year Project.\n\nIt uses Google MediaPipe for 3D hand tracking, a Random Forest Classifier for static key-pose recognition, and Gemini 1.5 Flash for natural sentence generation.", style: TextStyle(color: subtitleColor)),
                      actions: [TextButton(child: Text("Close", style: TextStyle(color: accentColor)), onPressed: () => Navigator.pop(context))],
                    )
                  );
                }, 
                child: Text("About", style: TextStyle(color: subtitleColor))
              ),
              SizedBox(width: 8),
              TextButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      backgroundColor: panelColor,
                      title: Text("AI Models Pipeline", style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
                      content: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("1. Vision:", style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
                          Text("Google MediaPipe (126 coordinates)", style: TextStyle(color: subtitleColor)),
                          SizedBox(height: 8),
                          Text("2. Classifier:", style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
                          Text("Scikit-Learn Random Forest", style: TextStyle(color: subtitleColor)),
                          SizedBox(height: 8),
                          Text("3. NLP:", style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
                          Text("Gemini 1.5 Flash", style: TextStyle(color: subtitleColor)),
                        ]
                      ),
                      actions: [TextButton(child: Text("Close", style: TextStyle(color: accentColor)), onPressed: () => Navigator.pop(context))],
                    )
                  );
                }, 
                child: Text("Models", style: TextStyle(color: subtitleColor))
              ),
              SizedBox(width: 8),
              Container(
                decoration: BoxDecoration(
                  color: Colors.redAccent.withOpacity(0.1),
                  border: Border.all(color: Colors.redAccent.withOpacity(0.5)),
                  borderRadius: BorderRadius.circular(8)
                ),
                child: Tooltip(
                  message: "Teach AI (On-Device Personalization)",
                  child: InkWell(
                    borderRadius: BorderRadius.circular(8),
                    onTap: () {
                    String newWord = "";
                    bool isLocalRecording = false;
                    showDialog(
                      context: context,
                      barrierDismissible: false,
                      builder: (context) {
                        return StatefulBuilder(
                          builder: (context, setLocalState) {
                            return AlertDialog(
                              backgroundColor: panelColor,
                              title: Text("Teach AI a New Sign 🧠", style: TextStyle(color: textColor)),
                              content: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  TextField(
                                    style: TextStyle(color: textColor),
                                    decoration: InputDecoration(
                                      hintText: "Enter the word (e.g. WATER)",
                                      hintStyle: TextStyle(color: subtitleColor),
                                      enabledBorder: OutlineInputBorder(borderSide: BorderSide(color: borderColor)),
                                      focusedBorder: OutlineInputBorder(borderSide: BorderSide(color: accentColor)),
                                    ),
                                    onChanged: (val) {
                                      setLocalState(() {
                                        newWord = val;
                                      });
                                    },
                                  ),
                                  SizedBox(height: 20),
                                  if (isLocalRecording)
                                    Column(
                                      children: [
                                        CircularProgressIndicator(color: accentColor),
                                        SizedBox(height: 10),
                                        Text("Recording...", style: TextStyle(color: subtitleColor)),
                                      ],
                                    )
                                  else
                                    Text("Hold the gesture in front of the camera, then click Record.", 
                                      style: TextStyle(color: subtitleColor, fontSize: 12),
                                      textAlign: TextAlign.center,
                                    ),
                                ],
                              ),
                              actions: [
                                TextButton(
                                  child: Text("Cancel", style: TextStyle(color: subtitleColor)),
                                  onPressed: () => Navigator.pop(context),
                                ),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(backgroundColor: accentColor),
                                  child: Text(isLocalRecording ? "Recording..." : "Record", style: TextStyle(color: Colors.white)),
                                  onPressed: isLocalRecording || newWord.trim().isEmpty ? null : () async {
                                    setLocalState(() => isLocalRecording = true);
                                    frameTimer?.cancel();
                                    await Future.delayed(Duration(milliseconds: 300));
                                    List<String> frames = [];
                                    for (int i = 0; i < 4; i++) {
                                      if (controller != null && controller!.value.isInitialized) {
                                        try {
                                          final image = await controller!.takePicture();
                                          final bytes = await image.readAsBytes();
                                          frames.add(base64Encode(bytes));
                                        } catch (e) {}
                                      }
                                      await Future.delayed(Duration(milliseconds: 250));
                                    }
                                    try {
                                      final response = await http.post(
                                        Uri.  parse('http://127.0.0.1:8000/teach_ai'),
                                        headers: {"Content-Type": "application/json"},
                                        body: jsonEncode({"word": newWord, "frames_base64": frames}),
                                      );
                                      Navigator.pop(context);
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(content: Text(response.statusCode == 200 ? "Successfully taught AI!" : "Failed to teach AI."))
                                      );
                                    } catch (e) {
                                      Navigator.pop(context);
                                      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Network error: $e")));
                                    }
                                    startFrameCapture();
                                  },
                                ),
                              ],
                            );
                          },
                        );
                      },
                    );
                  },
                  child: Padding(
                      padding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text("🧠", style: TextStyle(fontSize: 14)),
                          SizedBox(width: 6),
                          Text("Teach AI", style: TextStyle(color: isDarkMode ? Colors.white : Colors.black87, fontWeight: FontWeight.w600, fontSize: 13)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(width: 12),
              Tooltip(
                message: isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode",
                child: IconButton(
                  icon: Icon(isDarkMode ? Icons.light_mode : Icons.dark_mode_outlined, color: subtitleColor),
                  onPressed: () {
                    setState(() {
                      isDarkMode = !isDarkMode;
                    });
                  },
                ),
              ),
            ],
          )
        ],
      ),
    );
  }

  Widget buildCameraSection() {
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(Icons.videocam_outlined, color: accentColor, size: 20),
                  SizedBox(width: 8),
                  Text("Video Capture", style: TextStyle(color: textColor, fontSize: 16, fontWeight: FontWeight.w600)),
                ],
              ),
              Row(
                children: [
                  Text(
                    isRecording ? "LIVE" : "OFF",
                    style: TextStyle(
                      color: isRecording ? const Color(0xFF00E676) : subtitleColor,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(width: 8),
                  Switch(
                    value: isRecording,
                    activeColor: const Color(0xFF00E676),
                    onChanged: (v) => toggleRecording(),
                  ),
                ],
              )
            ],
          ),
          SizedBox(height: 10),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Container(
                width: double.infinity,
                color: Colors.black,
                child: (controller != null && controller!.value.isInitialized)
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          Transform.scale(
                            scaleX: isMirrored ? -1 : 1,
                            child: CameraPreview(controller!),
                          ),
                          if (isRecording && currentLandmarks.isNotEmpty)
                            CustomPaint(
                              painter: HandLandmarkPainter(
                                landmarks: currentLandmarks,
                                isDark: isDarkMode,
                                isMirrored: isMirrored,
                              ),
                            ),
                          if (!isRecording)
                            Container(
                              color: Colors.black.withOpacity(0.7),
                              child: Center(
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.videocam_off_outlined, color: Colors.white54, size: 44),
                                    SizedBox(height: 10),
                                    Text(
                                      "Video Capture Inactive",
                                      style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
                                    ),
                                    SizedBox(height: 4),
                                    Text(
                                      "Toggle the switch above to start real-time detection",
                                      style: TextStyle(color: Colors.white60, fontSize: 12),
                                    ),
                                  ],
                                ),
                              ),
                            )
                          else if (currentLandmarks.isNotEmpty)
                            Positioned(
                              top: 10,
                              left: 10,
                              child: Container(
                                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                decoration: BoxDecoration(
                                  color: Colors.black.withOpacity(0.65),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: const Color(0xFF00E676).withOpacity(0.6)),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      width: 8,
                                      height: 8,
                                      decoration: BoxDecoration(color: const Color(0xFF00E676), shape: BoxShape.circle),
                                    ),
                                    SizedBox(width: 6),
                                    Text(
                                      "TRACKING (${currentLandmarks.length} HAND${currentLandmarks.length > 1 ? "S" : ""})",
                                      style: TextStyle(color: const Color(0xFF00E676), fontSize: 11, fontWeight: FontWeight.bold),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                        ],
                      )
                    : Center(child: CircularProgressIndicator(color: accentColor)),
              ),
            ),
          ),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Tooltip(
                message: "Mirror Camera Feed",
                child: OutlinedButton.icon(
                  icon: Icon(isMirrored ? Icons.flip : Icons.crop_free, size: 15),
                  label: Text(isMirrored ? "Mirrored" : "Normal"),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: subtitleColor,
                    side: BorderSide(color: borderColor),
                    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
                  ),
                  onPressed: () => setState(() => isMirrored = !isMirrored),
                ),
              ),
              Row(
                children: [
                  Icon(Icons.bolt, size: 14, color: isRecording ? const Color(0xFF00E676) : subtitleColor),
                  SizedBox(width: 4),
                  Text(
                    isRecording ? "30 FPS Pipeline" : "Idle",
                    style: TextStyle(color: subtitleColor, fontSize: 12),
                  ),
                ],
              ),
            ],
          )
        ],
      ),
    );
  }

  Widget buildDetectionSection() {
    bool isOff = !isRecording;
    bool isNoHand = currentDetection.toLowerCase().contains("no hand");
    bool isAwaiting = currentDetection.toLowerCase().contains("awaiting") || currentDetection.toLowerCase().contains("tracking") || currentDetection.toLowerCase().contains("camera off");
    bool hasDetection = !isOff && !isNoHand && !isAwaiting;

    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Current Detection", style: TextStyle(color: textColor, fontSize: 16, fontWeight: FontWeight.w600)),
              Container(
                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: isOff 
                      ? Colors.grey.withOpacity(0.12) 
                      : (hasDetection ? const Color(0xFF00E676).withOpacity(0.12) : Colors.orangeAccent.withOpacity(0.12)),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: isOff 
                        ? Colors.grey.withOpacity(0.3) 
                        : (hasDetection ? const Color(0xFF00E676).withOpacity(0.4) : Colors.orangeAccent.withOpacity(0.4)),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: isOff 
                            ? Colors.grey 
                            : (hasDetection ? const Color(0xFF00E676) : Colors.orangeAccent),
                        shape: BoxShape.circle,
                      ),
                    ),
                    SizedBox(width: 5),
                    Text(
                      isOff ? "INACTIVE" : (hasDetection ? "DETECTED" : "SEARCHING"),
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: isOff 
                            ? subtitleColor 
                            : (hasDetection ? const Color(0xFF00E676) : Colors.orangeAccent),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          Expanded(
            child: Center(
              child: FittedBox(
                fit: BoxFit.scaleDown,
                child: Text(
                  hasDetection ? currentDetection.toUpperCase() : currentDetection,
                  style: TextStyle(
                    color: hasDetection 
                        ? (isDarkMode ? Color(0xFF8C9EFF) : Color(0xFF304FFE)) 
                        : subtitleColor,
                    fontSize: hasDetection ? 32 : 18,
                    fontWeight: hasDetection ? FontWeight.bold : FontWeight.w500,
                    fontStyle: hasDetection ? FontStyle.normal : FontStyle.italic,
                    letterSpacing: hasDetection ? 2.0 : 0.5,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildTranslationSection() {
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Translation", style: TextStyle(color: textColor, fontSize: 16, fontWeight: FontWeight.w600)),
              IconButton(
                padding: EdgeInsets.zero,
                constraints: BoxConstraints(),
                tooltip: "Speak Sentence",
                icon: Icon(Icons.volume_up, color: subtitleColor, size: 20), 
                onPressed: generateSentence
              ),
            ],
          ),
          SizedBox(height: 8),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: inputBgColor,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: borderColor),
              ),
              child: SingleChildScrollView(
                child: Text(
                  displaySentence,
                  style: TextStyle(color: subtitleColor, fontSize: 15),
                ),
              ),
            ),
          ),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                icon: Icon(isRecording ? Icons.stop : Icons.play_arrow, size: 16),
                label: Text(isRecording ? "Stop" : "Start"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: isRecording ? Color(0xFFB71C1C) : accentColor,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  padding: EdgeInsets.symmetric(horizontal: 18, vertical: 8),
                ),
                onPressed: toggleRecording,
              ),
              SizedBox(width: 10),
              OutlinedButton(
                child: Text("Clear"),
                style: OutlinedButton.styleFrom(
                  foregroundColor: subtitleColor,
                  side: BorderSide(color: borderColor),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  padding: EdgeInsets.symmetric(horizontal: 18, vertical: 8),
                ),
                onPressed: clearSentence,
              ),
            ],
          ),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Detected Signs:", style: TextStyle(color: textColor, fontSize: 13, fontWeight: FontWeight.w600)),
              if (currentWords.isNotEmpty)
                Text("${currentWords.length} collected", style: TextStyle(color: subtitleColor, fontSize: 11)),
            ],
          ),
          SizedBox(height: 6),
          if (currentWords.isEmpty)
            Container(
              width: double.infinity,
              padding: EdgeInsets.symmetric(vertical: 8, horizontal: 10),
              decoration: BoxDecoration(
                color: inputBgColor.withOpacity(0.5),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: borderColor.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  Icon(Icons.info_outline, size: 14, color: subtitleColor.withOpacity(0.5)),
                  SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      "No signs detected yet. Signs will accumulate here as chips.",
                      style: TextStyle(color: subtitleColor.withOpacity(0.5), fontSize: 12, fontStyle: FontStyle.italic),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            )
          else
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: currentWords.asMap().entries.map((entry) {
                int idx = entry.key;
                String word = entry.value;
                return Container(
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: accentColor.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: accentColor.withOpacity(0.3)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        word.toLowerCase(),
                        style: TextStyle(
                          color: isDarkMode ? Color(0xFF8C9EFF) : Color(0xFF304FFE),
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      SizedBox(width: 4),
                      InkWell(
                        onTap: () => removeWord(idx),
                        child: Icon(Icons.close, size: 12, color: subtitleColor),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
        ],
      ),
    );
  }

  Widget buildHearingSection() {
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(Icons.mic, color: accentColor, size: 18),
                  SizedBox(width: 6),
                  Text("Voice to Text", style: TextStyle(color: textColor, fontSize: 16, fontWeight: FontWeight.w600)),
                ],
              ),
              InkWell(
                onTap: _listenToSpeech,
                borderRadius: BorderRadius.circular(20),
                child: Container(
                  padding: EdgeInsets.all(6),
                  decoration: BoxDecoration(
                    color: _isListening ? Colors.redAccent.withOpacity(0.2) : accentColor.withOpacity(0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    _isListening ? Icons.mic : Icons.mic_none, 
                    color: _isListening ? Colors.redAccent : accentColor,
                    size: 20,
                  ),
                ),
              )
            ],
          ),
          SizedBox(height: 8),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: inputBgColor,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: borderColor),
              ),
              child: SingleChildScrollView(
                child: Text(
                  _hearingPersonResponse,
                  style: TextStyle(
                    color: _isListening ? textColor : subtitleColor,
                    fontSize: 20, // Increased for high readability and accessibility
                    fontWeight: FontWeight.w500,
                    height: 1.35,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Column(
          children: [
            buildNavbar(),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(left: 20.0, right: 20.0, bottom: 16.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Left Column (Camera)
                    Expanded(
                      flex: 6,
                      child: buildCameraSection(),
                    ),
                    SizedBox(width: 16),
                    // Right Column (Details)
                    Expanded(
                      flex: 4,
                      child: Column(
                        children: [
                          Expanded(flex: 2, child: buildDetectionSection()),
                          SizedBox(height: 12),
                          Expanded(flex: 4, child: buildTranslationSection()),
                          SizedBox(height: 12),
                          Expanded(flex: 2, child: buildHearingSection()),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class HandLandmarkPainter extends CustomPainter {
  final List<List<Offset>> landmarks;
  final bool isDark;
  final bool isMirrored;

  HandLandmarkPainter({
    required this.landmarks,
    required this.isDark,
    this.isMirrored = true,
  });

  static const List<List<int>> connections = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle
    [9, 10], [10, 11], [11, 12],
    // Ring
    [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm knuckle base
    [5, 9], [9, 13], [13, 17],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty) return;

    final linePaint = Paint()
      ..color = const Color(0xFF00E676).withOpacity(0.9)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final glowPaint = Paint()
      ..color = const Color(0xFF00E676).withOpacity(0.3)
      ..strokeWidth = 5.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final jointOuterPaint = Paint()
      ..color = const Color(0xFFFFD600)
      ..style = PaintingStyle.fill;

    final jointInnerPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    final fingertipPaint = Paint()
      ..color = const Color(0xFFFF1744)
      ..style = PaintingStyle.fill;

    const fingertips = {4, 8, 12, 16, 20};

    for (final hand in landmarks) {
      if (hand.length < 21) continue;

      // Draw bone connections with glow
      for (final conn in connections) {
        final p1 = Offset(hand[conn[0]].dx * size.width, hand[conn[0]].dy * size.height);
        final p2 = Offset(hand[conn[1]].dx * size.width, hand[conn[1]].dy * size.height);
        canvas.drawLine(p1, p2, glowPaint);
        canvas.drawLine(p1, p2, linePaint);
      }

      // Draw joint dots
      for (int i = 0; i < hand.length; i++) {
        final pt = Offset(hand[i].dx * size.width, hand[i].dy * size.height);
        if (fingertips.contains(i)) {
          canvas.drawCircle(pt, 5.5, jointInnerPaint);
          canvas.drawCircle(pt, 4.0, fingertipPaint);
        } else if (i == 0) {
          // Wrist anchor
          canvas.drawCircle(pt, 6.5, jointOuterPaint);
          canvas.drawCircle(pt, 3.5, jointInnerPaint);
        } else {
          canvas.drawCircle(pt, 4.0, jointOuterPaint);
          canvas.drawCircle(pt, 2.0, jointInnerPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant HandLandmarkPainter oldDelegate) => true;
}
