// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import type { FileUploadProps } from "@cloudscape-design/components/file-upload";

// Shared i18n strings for Cloudscape FileUpload (used on every page).
export const fileUploadI18n: FileUploadProps.I18nStrings = {
  uploadButtonText: (multiple) => (multiple ? "Choose files" : "Choose file"),
  dropzoneText: (multiple) =>
    multiple ? "Drop files to upload" : "Drop file to upload",
  removeFileAriaLabel: (i) => `Remove file ${i + 1}`,
  limitShowFewer: "Show fewer files",
  limitShowMore: "Show more files",
  errorIconAriaLabel: "Error",
};
