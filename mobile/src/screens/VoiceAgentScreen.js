// ... imports
import { useState, useEffect, useRef } from "react";
import { View, Text, StyleSheet, Alert, ScrollView, Pressable } from "react-native";
import { Audio } from "expo-av";
import * as FileSystem from "expo-file-system/legacy";
import { theme } from "../utils/theme";
import ScreenHeader from "../components/ScreenHeader";
import Card from "../components/Card";
import PrimaryButton from "../components/PrimaryButton";

const API_URL = "http://172.18.226.85:5000/api"; 

const LANGUAGES = [
    { name: "English", code: "en" },
    { name: "Hindi", code: "hi" },
    { name: "Bengali", code: "bn" },
    { name: "Punjabi", code: "pa" },
    { name: "Tamil", code: "ta" },
    { name: "Telugu", code: "te" },
    { name: "Marathi", code: "mr" },
    { name: "Gujarati", code: "gu" },
    { name: "Kannada", code: "kn" },
    { name: "Malayalam", code: "ml" },
    { name: "Urdu", code: "ur" }
];

const UploadTypeMultipart = FileSystem.FileSystemUploadType 
  ? FileSystem.FileSystemUploadType.MULTIPART 
  : FileSystem.UploadType 
    ? FileSystem.UploadType.MULTIPART 
    : 0; 

const VoiceAgentScreen = ({ onBack, onSubmit }) => {
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [history, setHistory] = useState([]);
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [displayedText, setDisplayedText] = useState("");
  const [sound, setSound] = useState(null);
  const [callStatus, setCallStatus] = useState("disconnected"); // disconnected, connecting, active

  useEffect(() => {
    return () => {
      if (recording) recording.stopAndUnloadAsync();
      if (sound) sound.unloadAsync();
    };
  }, []);

  const playBase64Audio = async (base64String) => {
      if (!base64String) return;
      try {
          if (sound) {
              await sound.unloadAsync();
          }
          const uri = FileSystem.cacheDirectory + 'response.mp3';
          await FileSystem.writeAsStringAsync(uri, base64String, {
              encoding: FileSystem.EncodingType.Base64,
          });
          const { sound: newSound } = await Audio.Sound.createAsync({ uri });
          setSound(newSound);
          await newSound.playAsync();
      } catch (error) {
          console.error("Failed to play audio:", error);
      }
  };

  const startSession = async (lang) => {
      setCallStatus("connecting");
      try {
          const response = await fetch(`${API_URL}/interactions/voice/start`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ language: lang.name }) 
          });
          const data = await response.json();
          setSessionId(data.session_id);
          setDisplayedText(data.message);
          setHistory([{ role: 'assistant', content: data.message }]);
          setCallStatus("active");
          
          if (data.audio_base64) {
              playBase64Audio(data.audio_base64);
          }
      } catch (e) {
          console.error("Start session failed", e);
          Alert.alert("Error", "Could not connect to the agent. Please try again.");
          setCallStatus("disconnected");
      }
  };

  const startRecording = async () => {
    try {
      if (sound) await sound.stopAsync();
      
      const permission = await Audio.requestPermissionsAsync();
      if (permission.status === "granted") {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
        });

        const { recording } = await Audio.Recording.createAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY
        );
        setRecording(recording);
        setIsRecording(true);
      } else {
        Alert.alert("Permission denied", "We need microphone access.");
      }
    } catch (err) {
      console.error("Failed to start recording", err);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    setIsRecording(false);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);
    uploadAudio(uri);
  };

  const uploadAudio = async (uri) => {
    setProcessing(true);
    try {
      const historyJson = JSON.stringify(history);

      const uploadResult = await FileSystem.uploadAsync(
        `${API_URL}/interactions/voice`,
        uri,
        {
          fieldName: "audio",
          httpMethod: "POST",
          uploadType: UploadTypeMultipart,
          headers: {
            Accept: "application/json",
          },
          parameters: {
              session_id: sessionId || "",
              history: historyJson,
              language: selectedLanguage.name
          }
        }
      );

      if (uploadResult.status === 200) {
        const data = JSON.parse(uploadResult.body);
        
        const newHistory = [...history];
        if (data.transcript) {
            newHistory.push({ role: 'user', content: data.transcript });
        }
        if (data.response_text) {
            let cleanText = data.response_text;
            let isFinished = false;
            
            if (cleanText.includes("[FINISH]")) {
                cleanText = cleanText.replace("[FINISH]", "").trim();
                isFinished = true;
            }
            
            newHistory.push({ role: 'assistant', content: cleanText });
            setDisplayedText(cleanText);

            if (isFinished) {
                // Optionally auto-end after 3 seconds to let audio play
                setTimeout(() => {
                    endSession();
                }, 4000);
            }
        }
        setHistory(newHistory);

        if (data.audio_base64) {
            playBase64Audio(data.audio_base64);
        }
      }
    } catch (error) {
      console.error("Upload error:", error);
    } finally {
      setProcessing(false);
    }
  };

  const endSession = async () => {
      console.log("Ending session...", sessionId);
      setProcessing(true);
      try {
          const response = await fetch(`${API_URL}/interactions/voice/end`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                  session_id: sessionId,
                  history: history 
              }) 
          });
          console.log("End session status:", response.status);
          const data = await response.json();
          console.log("End session data:", data);
          if (data.success) {
              Alert.alert("Call Ended", `Your complaint has been filed. ID: ${data.data.complaint_number}`);
              onBack();
          } else {
              Alert.alert("Error", "Server failed to finalize complaint.");
          }
      } catch (e) {
          console.error("End session error:", e);
          Alert.alert("Error", "Failed to connect to server for finalization.");
      } finally {
          setProcessing(false);
      }
  };

  if (callStatus === "disconnected") {
      return (
          <View style={styles.container}>
              <ScreenHeader title="Voice Agent" onBack={onBack} />
              <View style={styles.langOverlay}>
                  <Text style={styles.langTitle}>Select your language</Text>
                  <ScrollView contentContainerStyle={styles.langGrid}>
                      {LANGUAGES.map((lang) => (
                          <Pressable 
                            key={lang.code} 
                            onPress={() => {
                                setSelectedLanguage(lang);
                                startSession(lang);
                            }}
                            style={styles.langItem}
                          >
                              <View style={styles.langCircle}>
                                  <Text style={styles.langInitial}>{lang.name[0]}</Text>
                              </View>
                              <Text style={styles.langName}>{lang.name}</Text>
                          </Pressable>
                      ))}
                  </ScrollView>
              </View>
          </View>
      );
  }

  return (
    <View style={styles.container}>
      <ScreenHeader
        title="Voice Agent"
        subtitle={selectedLanguage ? `Calling in ${selectedLanguage.name}` : "Connecting..."}
        onBack={callStatus === "connecting" ? onBack : null} // Prevent back during call unless connecting
      />
      
      <Card style={styles.chatCard}>
        {callStatus === "connecting" ? (
            <Text style={styles.loadingText}>Connecting to Agent...</Text>
        ) : (
            <ScrollView contentContainerStyle={styles.chatContent}>
                <Text style={styles.agentText}>{displayedText}</Text>
            </ScrollView>
        )}
      </Card>

      {callStatus === "active" && (
          <View style={styles.controls}>
            <View style={styles.statusRow}>
                 <Text style={styles.statusText}>{isRecording ? "Listening..." : processing ? "Thinking..." : "Tap to Speak"}</Text>
            </View>
            
            <PrimaryButton
              label={isRecording ? "Stop & Send" : "Speak to Agent"}
              onPress={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "accent" : "primary"}
              disabled={processing}
              style={styles.mainButton}
            />

            <PrimaryButton
              label="End Call"
              onPress={endSession}
              variant="secondary"
              disabled={processing}
            />
          </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    gap: 16,
  },
  langOverlay: {
      flex: 1,
      padding: 20,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.radius.xl,
      marginTop: 20,
  },
  langTitle: {
      fontSize: 20,
      fontWeight: '700',
      color: theme.colors.ink,
      marginBottom: 20,
      textAlign: 'center',
  },
  langGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12,
      justifyContent: 'center',
  },
  langItem: {
      width: '45%',
      aspectRatio: 1,
      backgroundColor: theme.colors.soft,
      borderRadius: theme.radius.lg,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: theme.colors.border,
  },
  langCircle: {
      width: 50,
      height: 50,
      borderRadius: 25,
      backgroundColor: theme.colors.primary,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: 8,
  },
  langInitial: {
      color: '#fff',
      fontSize: 20,
      fontWeight: '700',
  },
  langName: {
      fontSize: 14,
      color: theme.colors.ink,
      fontWeight: '600',
  },
  chatCard: {
    flex: 1,
    backgroundColor: theme.colors.surface,
    padding: theme.spacing.lg,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: theme.radius.xl,
    borderWidth: 2,
    borderColor: theme.colors.softAlt,
  },
  chatContent: {
      flexGrow: 1,
      justifyContent: 'center',
      alignItems: 'center',
  },
  agentText: {
      fontSize: 26,
      fontWeight: '600',
      color: theme.colors.primary,
      textAlign: 'center',
      lineHeight: 36,
  },
  loadingText: {
      fontSize: 18,
      color: theme.colors.inkMuted,
      fontStyle: 'italic',
  },
  controls: {
      gap: 12,
      paddingBottom: 20,
  },
  statusRow: {
      alignItems: 'center',
      marginBottom: 8,
  },
  statusText: {
      color: theme.colors.inkMuted,
      fontSize: 14,
      textTransform: "uppercase",
      letterSpacing: 2,
      fontWeight: "800"
  },
  mainButton: {
      height: 70,
      borderRadius: 35,
  }
});

export default VoiceAgentScreen;
