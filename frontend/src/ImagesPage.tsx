// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Table from "@cloudscape-design/components/table";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Popover from "@cloudscape-design/components/popover";
import Link from "@cloudscape-design/components/link";
import Badge from "@cloudscape-design/components/badge";
import StatusIndicator from "@cloudscape-design/components/status-indicator";

import {
  buildImage,
  buildRelease,
  checkImageUpdates,
  listImages,
  resetTierVerifications,
  type ImageTierStatus,
  type UpdateCheck,
} from "./api";
import { useNotify, errText } from "./notifications";

// Small "(i)" info link with an explanatory popover, used beside the action buttons.
function InfoLink({ title, body }: { title: string; body: string }) {
  return (
    <Popover header={title} content={body} triggerType="custom" dismissButton={false} position="top">
      <Link variant="info">info</Link>
    </Popover>
  );
}

const REBUILD_INFO =
  "Re-runs the CodeBuild project that builds THIS tier's Docker image and pushes it to ECR " +
  "(~10–15 min). Use it after the container code (entrypoint, eval.py, Dockerfile) changes so " +
  "training/eval jobs pick up the new bits. It rebuilds the same tag — it does NOT change which " +
  "LLaMA-Factory version the tier uses. After it succeeds, Reset verifications so models re-prove " +
  "on the new image.";

const RESET_INFO =
  "Clears every model's verified/incompatible record FOR THIS TIER, setting them back to untested. " +
  "Touches only the verification store — never deletes the image, models, or data. Verification is " +
  "tied to the exact image bits, so a 'verified' from before a rebuild is a stale claim; resetting " +
  "makes models re-prove themselves (via a run or the catalog's verify button). The shipped " +
  "known-good baseline survives a reset. Disabled when nothing is verified on this tier yet.";

// Map a CodeBuild build status to a Cloudscape status indicator.
function buildIndicator(status: string | null | undefined) {
  if (!status) return <Box color="text-status-inactive">never built</Box>;
  if (status === "SUCCEEDED") return <StatusIndicator type="success">succeeded</StatusIndicator>;
  if (status === "IN_PROGRESS") return <StatusIndicator type="in-progress">building…</StatusIndicator>;
  if (status === "FAILED" || status === "FAULT" || status === "TIMED_OUT" || status === "STOPPED")
    return <StatusIndicator type="error">{status.toLowerCase()}</StatusIndicator>;
  return <StatusIndicator type="pending">{status.toLowerCase()}</StatusIndicator>;
}

export function ImagesPage() {
  const [tiers, setTiers] = useState<ImageTierStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [updates, setUpdates] = useState<UpdateCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const { notify } = useNotify();

  function refresh() {
    setLoading(true);
    listImages()
      .then((r) => setTiers(r.tiers))
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }

  useEffect(refresh, []);

  async function checkUpdates() {
    setChecking(true);
    try {
      setUpdates(await checkImageUpdates());
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setChecking(false);
    }
  }

  async function buildNewRelease(lfTag: string) {
    setBusy((b) => ({ ...b, [`release:${lfTag}`]: true }));
    try {
      const r = await buildRelease(lfTag);
      notify({
        type: "info",
        content:
          `Building LLaMA-Factory ${lfTag} as new tier '${r.tier}' (CodeBuild ${r.buildId}). ` +
          `~10–15 min. When it succeeds the tier is usable immediately — every model on it starts ` +
          `untested, so smoke-test the ones you want (and check the Model catalog for newly-supported models).`,
      });
      setUpdates(null);
      setTimeout(refresh, 4000);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setBusy((b) => ({ ...b, [`release:${lfTag}`]: false }));
    }
  }

  async function rebuild(tier: string) {
    setBusy((b) => ({ ...b, [`build:${tier}`]: true }));
    try {
      const r = await buildImage(tier);
      notify({
        type: "info",
        content:
          `Started building the '${tier}' image (CodeBuild ${r.buildId}). This takes ~10–15 min; ` +
          `refresh to watch the build status. After it succeeds, reset verifications so models re-prove on the new bits.`,
      });
      setTimeout(refresh, 3000); // let the build register, then show IN_PROGRESS
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setBusy((b) => ({ ...b, [`build:${tier}`]: false }));
    }
  }

  async function resetVerifs(tier: string) {
    setBusy((b) => ({ ...b, [`reset:${tier}`]: true }));
    try {
      const r = await resetTierVerifications(tier);
      notify({
        type: "success",
        content:
          `Cleared ${r.cleared} verification record(s) for '${tier}'. Models on this tier are now untested ` +
          `until re-proved (launch a run or use the Model catalog's verify buttons).`,
      });
      refresh();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setBusy((b) => ({ ...b, [`reset:${tier}`]: false }));
    }
  }

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="The Docker image tiers the platform trains on. Each tier maps to one ECR image tag built from a specific LLaMA-Factory release. Old models stay on 'stable'; models that need a newer stack run on 'latest'. Rebuild a tier's image here, then reset its verifications so every model re-proves itself on the new bits."
          actions={
            <SpaceBetween direction="horizontal" size="xs">
              <Button iconName="status-info" onClick={checkUpdates} loading={checking}>
                Check for updates
              </Button>
              <Button iconName="refresh" onClick={refresh} loading={loading}>
                Refresh
              </Button>
            </SpaceBetween>
          }
        >
          Images (ECR tiers)
        </Header>
      }
    >
      <SpaceBetween size="m">
        {updates && (
          updates.newReleases.length > 0 ? (
            <Alert
              type="info"
              header={`Newer LLaMA-Factory release${updates.newReleases.length > 1 ? "s" : ""} available`}
              dismissible
              onDismiss={() => setUpdates(null)}
            >
              <SpaceBetween size="xs">
                <Box>
                  You build {updates.builtTags.join(", ")}. Newer on Docker Hub:{" "}
                  {updates.newReleases.join(", ")}. Building one adds it as a NEW tier — your existing
                  models stay on their current image; nothing is auto-upgraded.
                </Box>
                <SpaceBetween direction="horizontal" size="xs">
                  {updates.newReleases.map((tag) => (
                    <Button
                      key={tag}
                      variant="primary"
                      loading={!!busy[`release:${tag}`]}
                      onClick={() => buildNewRelease(tag)}
                    >
                      Build {tag}
                    </Button>
                  ))}
                </SpaceBetween>
              </SpaceBetween>
            </Alert>
          ) : (
            <Alert type="success" dismissible onDismiss={() => setUpdates(null)}>
              Up to date — you build the newest LLaMA-Factory release ({updates.newest ?? "n/a"}).
            </Alert>
          )
        )}
        <Table
          variant="container"
          loading={loading}
          items={tiers}
          trackBy="tier"
          resizableColumns
          columnDefinitions={[
            {
              id: "tier",
              header: "Tier",
              cell: (t) => (
                <SpaceBetween size="xxs">
                  <Box variant="strong">{t.tier}</Box>
                  <Badge color="grey">{t.tag}</Badge>
                </SpaceBetween>
              ),
            },
            {
              id: "ecr",
              header: "In ECR",
              cell: (t) =>
                t.existsInEcr ? (
                  <SpaceBetween size="xxs">
                    <StatusIndicator type="success">present</StatusIndicator>
                    <Box fontSize="body-s" color="text-status-inactive">
                      {t.sizeMB ? `${t.sizeMB} MB` : ""}
                      {t.pushedAt ? ` · pushed ${t.pushedAt.slice(0, 19)}` : ""}
                    </Box>
                  </SpaceBetween>
                ) : (
                  <StatusIndicator type="warning">not built</StatusIndicator>
                ),
            },
            {
              id: "imageUri",
              header: "Image URI",
              cell: (t) => (
                <Box fontSize="body-s" color="text-status-inactive">
                  {t.imageUri}
                </Box>
              ),
            },
            {
              id: "verified",
              header: "Verified models",
              cell: (t) => `${t.verifiedModels}`,
            },
            {
              id: "lastBuild",
              header: "Last build",
              cell: (t) => buildIndicator(t.lastBuild?.status),
            },
            {
              id: "actions",
              header: "Actions",
              cell: (t) => (
                <SpaceBetween direction="horizontal" size="xs" alignItems="center">
                  <Button
                    loading={!!busy[`build:${t.tier}`]}
                    disabled={t.lastBuild?.status === "IN_PROGRESS"}
                    onClick={() => rebuild(t.tier)}
                  >
                    {t.existsInEcr ? "Rebuild" : "Build"}
                  </Button>
                  <InfoLink title={t.existsInEcr ? "Rebuild image" : "Build image"} body={REBUILD_INFO} />
                  <Button
                    variant="normal"
                    loading={!!busy[`reset:${t.tier}`]}
                    disabled={t.verifiedModels === 0}
                    onClick={() => resetVerifs(t.tier)}
                  >
                    Reset verifications
                  </Button>
                  <InfoLink title="Reset verifications" body={RESET_INFO} />
                </SpaceBetween>
              ),
            },
          ]}
          empty={<Box textAlign="center">No image tiers configured.</Box>}
        />
      </SpaceBetween>
    </ContentLayout>
  );
}
