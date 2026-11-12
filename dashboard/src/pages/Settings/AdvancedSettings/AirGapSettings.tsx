import { useCallback, useEffect, useState } from "react";
import { Alert, Button, Switch } from "antd";
import { WifiOff } from "lucide-react";
import { useTranslation } from "react-i18next";
import { message } from "@/utils/antdMessage";
import {
  octopSettingsApi,
  type OctopAirGapSettings,
} from "../../../api/modules/settings";
import { pluginsApi } from "../../../api/modules/plugins";
import { apiErrorMessage } from "../../../utils/apiError";
import { TabPanelHeader } from "./TabPanelHeader";
import styles from "./AirGapSettings.module.less";

/** Plugin that owns the air-gap runtime (helper / watchdog seeding). */
const EGRESS_PLUGIN_ID = "egress-guard";
/** While waiting for the helper, re-read the switch so "pending" can resolve itself. */
const PENDING_POLL_MS = 5000;

type PluginAvailability =
  | "ready"
  | "missing"
  | "disabled"
  | "not_loaded"
  | "unknown";

type StatusKind =
  | "on"
  | "pending"
  | "off"
  | "unavailable"
  | "loading"
  | "unknown";

function resolveAvailability(
  plugins: Awaited<ReturnType<typeof pluginsApi.list>>,
): PluginAvailability {
  const plugin = plugins.find((item) => item.id === EGRESS_PLUGIN_ID);
  if (!plugin) return "missing";
  if (plugin.enabled === false) return "disabled";
  if (plugin.loaded === false) return "not_loaded";
  return "ready";
}

export default function AirGapSettingsPanel() {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [state, setState] = useState<OctopAirGapSettings | null>(null);
  const [availability, setAvailability] =
    useState<PluginAvailability>("unknown");

  const load = useCallback(async () => {
    setLoading(true);
    const [settingsResult, pluginsResult] = await Promise.allSettled([
      octopSettingsApi.airgap(),
      pluginsApi.list(),
    ]);
    if (settingsResult.status === "fulfilled") {
      setState(settingsResult.value);
    } else {
      message.error(
        apiErrorMessage(
          settingsResult.reason,
          t("advancedSettings.airGap.loadFailed"),
          t,
        ),
      );
    }
    setAvailability(
      pluginsResult.status === "fulfilled"
        ? resolveAvailability(pluginsResult.value)
        : "unknown",
    );
    setLoading(false);
  }, [t]);

  useEffect(() => {
    void load();
  }, [load]);

  const supported = state?.supported !== false;
  const pluginUnavailable =
    supported && availability !== "ready" && availability !== "unknown";
  const pending = supported && state?.enabled === true && state.active !== true;

  // Refresh while the helper is being started so "pending" does not stay forever.
  useEffect(() => {
    if (!pending || pluginUnavailable) return;
    const timer = window.setInterval(() => {
      octopSettingsApi
        .airgap()
        .then(setState)
        .catch(() => undefined);
    }, PENDING_POLL_MS);
    return () => window.clearInterval(timer);
  }, [pending, pluginUnavailable]);

  const onToggle = async (checked: boolean) => {
    setSaving(true);
    try {
      const saved = await octopSettingsApi.putAirgap({ enabled: checked });
      setState(saved);
      message.success(
        t(
          checked
            ? "advancedSettings.airGap.enabledToast"
            : "advancedSettings.airGap.disabledToast",
        ),
      );
    } catch (err) {
      message.error(
        apiErrorMessage(err, t("advancedSettings.airGap.saveFailed"), t),
      );
    } finally {
      setSaving(false);
    }
  };

  const enabled = state?.enabled === true;
  const overridden = state?.source === "env" || state?.source === "switch_file";
  const busy = loading || saving;
  // When the plugin is unavailable the switch must not pretend it can turn the
  // air-gap on; turning a saved "on" back off stays allowed so the user can
  // restore networking without reinstalling anything.
  const switchDisabled =
    busy || state === null || !supported || (pluginUnavailable && !enabled);

  let statusKind: StatusKind;
  if (state === null) {
    statusKind = loading ? "loading" : "unknown";
  } else if (pluginUnavailable || !supported) {
    statusKind = "unavailable";
  } else if (!enabled) {
    statusKind = "off";
  } else {
    statusKind = state.active ? "on" : "pending";
  }

  const statusText = t(`advancedSettings.airGap.status.${statusKind}`);
  const statusClass =
    statusKind === "on"
      ? styles.statusOn
      : statusKind === "off"
        ? styles.statusOff
        : statusKind === "pending"
          ? styles.statusPending
          : styles.statusNeutral;

  const unavailableHintKey =
    availability === "missing"
      ? "advancedSettings.airGap.pluginMissing"
      : availability === "disabled"
        ? "advancedSettings.airGap.pluginDisabled"
        : availability === "not_loaded"
          ? "advancedSettings.airGap.pluginNotLoaded"
          : null;
  const unavailableHint =
    supported && unavailableHintKey ? t(unavailableHintKey) : null;

  return (
    <div className={styles.page}>
      <TabPanelHeader
        icon={<WifiOff size={20} />}
        title={t("advancedSettings.airGap.title")}
        description={t("advancedSettings.airGap.description")}
      />

      <div className={styles.stack}>
        <section className={styles.card}>
          <div className={styles.toggleRow}>
            <Switch
              checked={enabled}
              loading={saving}
              disabled={switchDisabled}
              onChange={(checked) => void onToggle(checked)}
            />
            <div className={styles.toggleCopy}>
              <p className={styles.toggleLabel}>
                {t("advancedSettings.airGap.toggleLabel")}
              </p>
              <p className={styles.toggleHint}>
                {t("advancedSettings.airGap.toggleHint")}
              </p>
              <p className={statusClass}>{statusText}</p>
            </div>
          </div>
        </section>

        {state === null && !loading ? (
          <Alert
            type="error"
            showIcon
            message={t("advancedSettings.airGap.loadFailed")}
            action={
              <Button size="small" onClick={() => void load()}>
                {t("advancedSettings.airGap.retry")}
              </Button>
            }
          />
        ) : null}

        {!supported ? (
          <Alert
            type="warning"
            showIcon
            message={t("advancedSettings.airGap.unsupported")}
          />
        ) : null}

        {unavailableHint ? (
          <Alert
            type="warning"
            showIcon
            message={unavailableHint}
            description={
              enabled
                ? t("advancedSettings.airGap.pluginEscapeHint")
                : undefined
            }
          />
        ) : null}

        {availability === "unknown" && supported ? (
          <Alert
            type="warning"
            showIcon
            message={t("advancedSettings.airGap.pluginCheckFailed")}
          />
        ) : null}

        {statusKind === "pending" ? (
          <Alert
            type="info"
            showIcon
            message={t("advancedSettings.airGap.pendingExplain")}
          />
        ) : null}

        {overridden ? (
          <Alert
            type="warning"
            showIcon
            message={t(
              state?.source === "env"
                ? "advancedSettings.airGap.overrideEnv"
                : "advancedSettings.airGap.overrideFile",
            )}
          />
        ) : null}

        <p className={styles.note}>{t("advancedSettings.airGap.effectHint")}</p>
        <p className={styles.note}>{t("advancedSettings.airGap.uacHint")}</p>
      </div>
    </div>
  );
}
