import sys

new_ui = """
  Widget buildNavbar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
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
                      content: Text("SignSense is an AI-powered Two-Way Indian Sign Language Translator designed for our Final Year Project.\\n\\nIt uses Google MediaPipe for 3D hand tracking, a Random Forest Classifier for static key-pose recognition, and Gemini 1.5 Flash for natural sentence generation.", style: TextStyle(color: subtitleColor)),
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
              SizedBox(width: 16),
              IconButton(
                icon: Icon(isDarkMode ? Icons.light_mode : Icons.dark_mode_outlined, color: subtitleColor),
                onPressed: () {
                  setState(() {
                    isDarkMode = !isDarkMode;
                  });
                },
              ),
            ],
          )
        ],
      ),
    );
  }

  Widget buildCameraSection() {
    return Container(
      padding: EdgeInsets.all(20),
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
              Text("Video Capture", style: TextStyle(color: textColor, fontSize: 18, fontWeight: FontWeight.w600)),
              Row(
                children: [
                  Text(isRecording ? "On" : "Off", style: TextStyle(color: subtitleColor, fontSize: 14)),
                  SizedBox(width: 8),
                  Switch(
                    value: isRecording,
                    activeColor: accentColor,
                    onChanged: (v) => toggleRecording(),
                  ),
                ],
              )
            ],
          ),
          SizedBox(height: 16),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Container(
                width: double.infinity,
                color: Colors.black,
                child: (controller != null && controller!.value.isInitialized)
                    ? CameraPreview(controller!)
                    : Center(child: CircularProgressIndicator(color: accentColor)),
              ),
            ),
          ),
          SizedBox(height: 16),
          Center(
            child: OutlinedButton.icon(
              icon: Icon(Icons.flip_camera_ios, size: 16),
              label: Text("Switch Camera"),
              style: OutlinedButton.styleFrom(
                foregroundColor: accentColor,
                side: BorderSide(color: accentColor.withOpacity(0.5)),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
              onPressed: () {},
            ),
          )
        ],
      ),
    );
  }

  Widget buildDetectionSection() {
    String currentWord = currentWords.isNotEmpty ? currentWords.last.toLowerCase() : "waiting...";
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Current Detection", style: TextStyle(color: textColor, fontSize: 18, fontWeight: FontWeight.w600)),
          Expanded(
            child: Center(
              child: Text(
                currentWord,
                style: TextStyle(color: textColor, fontSize: 32, fontWeight: FontWeight.bold, letterSpacing: 2.0),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildTranslationSection() {
    return Container(
      padding: EdgeInsets.all(20),
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
              Text("Translation", style: TextStyle(color: textColor, fontSize: 18, fontWeight: FontWeight.w600)),
              IconButton(icon: Icon(Icons.volume_up, color: subtitleColor), onPressed: generateSentence),
            ],
          ),
          SizedBox(height: 12),
          Container(
            width: double.infinity,
            height: 100,
            padding: EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: inputBgColor,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: borderColor),
            ),
            child: Text(
              displaySentence,
              style: TextStyle(color: subtitleColor, fontSize: 16),
            ),
          ),
          SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                icon: Icon(isRecording ? Icons.stop : Icons.play_arrow, size: 18),
                label: Text(isRecording ? "Stop" : "Start"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: isRecording ? Color(0xFFB71C1C) : accentColor,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                onPressed: toggleRecording,
              ),
              SizedBox(width: 16),
              OutlinedButton(
                child: Text("Clear"),
                style: OutlinedButton.styleFrom(
                  foregroundColor: subtitleColor,
                  side: BorderSide(color: borderColor),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                onPressed: clearSentence,
              ),
            ],
          ),
          SizedBox(height: 24),
          Text("Detected Signs:", style: TextStyle(color: textColor, fontSize: 14, fontWeight: FontWeight.w600)),
          SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: currentWords.map((word) {
              return Container(
                padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: accentColor.withOpacity(0.3)),
                ),
                child: Text(word.toLowerCase(), style: TextStyle(color: isDarkMode ? Color(0xFF8C9EFF) : Color(0xFF304FFE), fontSize: 14)),
              );
            }).toList(),
          )
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
                padding: const EdgeInsets.only(left: 24.0, right: 24.0, bottom: 24.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Left Column (Camera)
                    Expanded(
                      flex: 6,
                      child: buildCameraSection(),
                    ),
                    SizedBox(width: 24),
                    // Right Column (Details)
                    Expanded(
                      flex: 4,
                      child: Column(
                        children: [
                          Expanded(flex: 2, child: buildDetectionSection()),
                          SizedBox(height: 24),
                          Expanded(flex: 5, child: buildTranslationSection()),
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
"""

with open('lib/main.dart', 'a') as f:
    f.write(new_ui)
