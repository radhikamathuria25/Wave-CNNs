{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ImpWCNet.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ImpWCNet.labels" -}}
helm.sh/chart: {{ include "ImpWCNet.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
{{- end }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
    