import { useState } from "react";
import { View, Text, StyleSheet } from "react-native";
import { theme } from "../utils/theme";
import ScreenHeader from "../components/ScreenHeader";
import Card from "../components/Card";
import InputField from "../components/InputField";
import PrimaryButton from "../components/PrimaryButton";

const VoiceAgentScreen = ({ onBack, onSubmit }) => {
  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [summary, setSummary] = useState("");

  const handleConnect = () => {
    setConnected((prev) => !prev);
  };

  const handleGenerate = () => {
    setTranscript("Caller reports a large pothole near Gate 4, causing traffic delays.");
    setSummary("Pothole near Gate 4, high traffic impact. Assign to Roads Dept.");
  };

  const handleSubmit = () => {
    onSubmit({
      title: "Voice complaint",
      category: "Roads",
      summary: summary || "Summary pending.",
      transcript: transcript || "Transcript pending.",
    });
  };

  return (
    <View style={styles.container}>
      <ScreenHeader
        title="Talk to Agent"
        subtitle="Stream your voice and get an AI summary."
        onBack={onBack}
      />
      <Card style={styles.card}>
        <View style={styles.row}>
          <Text style={styles.label}>Connection</Text>
          <Text style={[styles.status, connected ? styles.statusOn : styles.statusOff]}>
            {connected ? "Connected" : "Offline"}
          </Text>
        </View>
        <PrimaryButton
          label={connected ? "Disconnect" : "Connect to AI agent"}
          onPress={handleConnect}
        />
        <PrimaryButton label="Start talking" variant="secondary" onPress={() => {}} />
      </Card>
      <Card style={styles.card}>
        <InputField
          label="Transcript (auto-filled)"
          value={transcript}
          onChangeText={setTranscript}
          placeholder="Live transcript appears here"
          multiline
        />
        <PrimaryButton label="Generate summary" variant="secondary" onPress={handleGenerate} />
        <InputField
          label="AI Summary"
          value={summary}
          onChangeText={setSummary}
          placeholder="AI summary appears here"
          multiline
        />
        <PrimaryButton label="Send summary to backend" onPress={handleSubmit} />
      </Card>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 16,
  },
  card: {
    gap: 12,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: theme.colors.ink,
  },
  status: {
    fontSize: 12,
    fontWeight: "700",
  },
  statusOn: {
    color: theme.colors.accent,
  },
  statusOff: {
    color: theme.colors.inkMuted,
  },
});

export default VoiceAgentScreen;
