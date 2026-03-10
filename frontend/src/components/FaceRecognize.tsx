import { useState, useCallback, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { ScanFace } from "lucide-react";
import WebcamCapture from "./WebcamCapture";
import { recognizeFaces, type RecognizedFace } from "@/lib/api";

export default function FaceRecognize() {
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<RecognizedFace[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleCapture = useCallback((blob: Blob) => {
    setImageBlob(blob);
    setPreview(URL.createObjectURL(blob));
    setResults(null);
    setError(null);
  }, []);

  const clearPreview = useCallback(() => {
    setImageBlob(null);
    setPreview(null);
    setResults(null);
  }, []);

  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !results) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    for (const face of results) {
      const [x1, y1, x2, y2] = face.bbox;
      const isKnown = face.name !== "unknown";
      ctx.strokeStyle = isKnown ? "#22c55e" : "#ef4444";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${face.name} (${Math.round(face.confidence * 100)}%)`;
      ctx.font = "bold 16px sans-serif";
      const metrics = ctx.measureText(label);
      const padding = 4;
      ctx.fillStyle = isKnown ? "#22c55e" : "#ef4444";
      ctx.fillRect(x1, y1 - 24, metrics.width + padding * 2, 24);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + padding, y1 - 7);
    }
  }, [results]);

  useEffect(() => {
    if (results && preview) {
      const img = new Image();
      img.onload = () => {
        imgRef.current = img;
        drawBoxes();
      };
      img.src = preview;
    }
  }, [results, preview, drawBoxes]);

  const handleRecognize = async () => {
    if (!imageBlob) return;

    setLoading(true);
    setError(null);
    try {
      const file = new File([imageBlob], "face.jpg", { type: "image/jpeg" });
      const data = await recognizeFaces(file);
      setResults(data.faces);
    } catch (err: unknown) {
      if (typeof err === "object" && err !== null && "response" in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        setError(axiosErr.response?.data?.detail ?? "Erreur de reconnaissance");
      } else {
        setError("Erreur de reconnaissance");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ScanFace className="h-5 w-5" />
          Reconnaitre des visages
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {results && preview ? (
          <div className="space-y-3">
            <canvas
              ref={canvasRef}
              className="w-full max-w-[480px] rounded-lg border border-border"
            />
            <div className="flex flex-wrap gap-2">
              {results.map((face, i) => (
                <Badge
                  key={i}
                  variant={face.name !== "unknown" ? "default" : "destructive"}
                >
                  {face.name} — {Math.round(face.confidence * 100)}%
                </Badge>
              ))}
            </div>
            <Button variant="outline" onClick={clearPreview}>
              Nouvelle image
            </Button>
          </div>
        ) : (
          <>
            <WebcamCapture
              onCapture={handleCapture}
              preview={preview}
              onClearPreview={clearPreview}
            />
            <Button
              onClick={handleRecognize}
              disabled={loading || !imageBlob}
              className="w-full"
            >
              {loading ? "Analyse..." : "Reconnaitre"}
            </Button>
          </>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
