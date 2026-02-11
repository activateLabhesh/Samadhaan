import { Pressable, StyleSheet, Text } from "react-native";
import { theme } from "../utils/theme";

const PrimaryButton = ({ label, onPress, variant = "primary", style }) => {
  const isPrimary = variant === "primary";
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.base,
        isPrimary ? styles.primary : styles.secondary,
        pressed && styles.pressed,
        style,
      ]}
    >
      <Text style={[styles.label, isPrimary ? styles.primaryLabel : styles.secondaryLabel]}>
        {label}
      </Text>
    </Pressable>
  );
};

const styles = StyleSheet.create({
  base: {
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 44,
  },
  primary: {
    backgroundColor: theme.colors.primary,
    shadowColor: "#111b2f",
    shadowOpacity: 0.2,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 2,
  },
  secondary: {
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  label: {
    fontWeight: "600",
    fontSize: 14,
  },
  primaryLabel: {
    color: "#fff",
  },
  secondaryLabel: {
    color: theme.colors.primary,
  },
  pressed: {
    opacity: 0.85,
  },
});

export default PrimaryButton;
