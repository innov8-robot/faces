import axios from "axios";

export const API_BASE = "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE,
});

export interface RegisteredFace {
  id: string;
  name: string;
}

export interface RecognizedFace {
  name: string;
  confidence: number;
  bbox: number[];
}

export async function registerFace(file: File, name: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  const { data } = await api.post("/faces/register", form);
  return data;
}

export async function recognizeFaces(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<{ faces: RecognizedFace[]; count: number }>(
    "/faces/recognize",
    form
  );
  return data;
}

export async function listFaces() {
  const { data } = await api.get<{ faces: RegisteredFace[] }>("/faces/");
  return data.faces;
}

export async function deleteFace(id: string) {
  const { data } = await api.delete(`/faces/${id}`);
  return data;
}

export async function getStreamFaces(source: string) {
  const { data } = await api.get<{ faces: RecognizedFace[]; active: boolean }>(
    "/stream/faces",
    { params: { source } }
  );
  return data;
}

export async function stopStream(source: string) {
  const { data } = await api.post("/stream/stop", null, { params: { source } });
  return data;
}

export function getMjpegUrl(source: string) {
  return `${API_BASE}/stream/mjpeg?source=${encodeURIComponent(source)}`;
}
