import { useRef, useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Video, VideoOff, Monitor, Bot } from "lucide-react";
import {
  recognizeFaces,
  getStreamFaces,
  stopStream,
  getMjpegUrl,
  type RecognizedFace,
} from "@/lib/api";

type SourceMode = "webcam" | "robot";

export default function LiveFeed() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopRef = useRef<number | null>(null);
  const processingRef = useRef(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [mode, setMode] = useState<SourceMode>("webcam");
  const [active, setActive] = useState(false);
  const [faces, setFaces] = useState<RecognizedFace[]>([]);

  // --- Webcam mode (browser camera, for PC) ---
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setActive(true);
    } catch {
      alert("Impossible d'acceder a la camera");
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (loopRef.current) {
      cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setActive(false);
    setFaces([]);
  }, []);

  const captureFrame = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        resolve(null);
        return;
      }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(video, 0, 0);
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.7);
    });
  }, []);

  const drawOverlay = useCallback((detectedFaces: RecognizedFace[]) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const face of detectedFaces) {
      const [x1, y1, x2, y2] = face.bbox;
      const isKnown = face.name !== "unknown" && face.name !== "inconnu";
      const color = isKnown ? "#22c55e" : "#ef4444";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${face.name} (${Math.round(face.confidence * 100)}%)`;
      ctx.font = "bold 18px sans-serif";
      const metrics = ctx.measureText(label);
      const pad = 6;

      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 28, metrics.width + pad * 2, 28);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + pad, y1 - 8);
    }
  }, []);

  // Webcam recognition loop
  useEffect(() => {
    if (!active || mode !== "webcam") return;

    let running = true;
    let lastSendTime = 0;
    const INTERVAL = 200; // ~5 fps

    const loop = async () => {
      if (!running) return;

      const now = Date.now();
      if (now - lastSendTime >= INTERVAL && !processingRef.current) {
        lastSendTime = now;
        processingRef.current = true;

        const blob = await captureFrame();
        if (blob) {
          try {
            const file = new File([blob], "frame.jpg", { type: "image/jpeg" });
            const data = await recognizeFaces(file);
            if (running) {
              setFaces(data.faces);
              drawOverlay(data.faces);
            }
          } catch {
            // skip
          }
        }
        processingRef.current = false;
      }

      if (running) {
        loopRef.current = requestAnimationFrame(loop);
      }
    };

    loopRef.current = requestAnimationFrame(loop);

    return () => {
      running = false;
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
    };
  }, [active, mode, captureFrame, drawOverlay]);

  // --- Robot mode (RealSense via backend MJPEG) ---
  const startRobot = useCallback(() => {
    setActive(true);
    // Poll detected faces from the backend stream
    pollRef.current = setInterval(async () => {
      try {
        const data = await getStreamFaces("4");
        setFaces(data.faces);
      } catch {
        // skip
      }
    }, 500);
  }, []);

  const stopRobot = useCallback(async () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    try {
      await stopStream("4");
    } catch {
      // ignore
    }
    setActive(false);
    setFaces([]);
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleStart = () => {
    if (mode === "webcam") startWebcam();
    else startRobot();
  };

  const handleStop = () => {
    if (mode === "webcam") stopWebcam();
    else stopRobot();
  };

  const switchMode = (newMode: SourceMode) => {
    if (active) handleStop();
    setMode(newMode);
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            Flux video live
          </CardTitle>
          <div className="flex items-center gap-2">
            {active ? (
              <Button variant="destructive" size="sm" onClick={handleStop}>
                <VideoOff className="mr-2 h-4 w-4" />
                Arreter
              </Button>
            ) : (
              <Button size="sm" onClick={handleStart}>
                <Video className="mr-2 h-4 w-4" />
                Demarrer
              </Button>
            )}
          </div>
        </div>

        {/* Source selector */}
        <div className="mt-3 flex items-center gap-2">
          <Button
            variant={mode === "webcam" ? "default" : "outline"}
            size="sm"
            onClick={() => switchMode("webcam")}
            disabled={active}
          >
            <Monitor className="mr-2 h-4 w-4" />
            Webcam locale
          </Button>
          <Button
            variant={mode === "robot" ? "default" : "outline"}
            size="sm"
            onClick={() => switchMode("robot")}
            disabled={active}
          >
            <Bot className="mr-2 h-4 w-4" />
            Camera robot
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="relative w-full overflow-hidden rounded-lg border border-border bg-black">
          {!active && (
            <div className="flex aspect-video items-center justify-center">
              <p className="text-muted-foreground">
                {mode === "webcam" ? "Camera inactive" : "Camera robot inactive"}
              </p>
            </div>
          )}

          {/* Webcam mode: browser video + canvas overlay */}
          {mode === "webcam" && (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full ${active ? "" : "hidden"}`}
              />
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full ${active ? "" : "hidden"}`}
              />
            </>
          )}

          {/* Robot mode: MJPEG from backend (RealSense, annotations baked in) */}
          {mode === "robot" && active && (
            <img
              src={getMjpegUrl("4")}
              alt="Robot camera stream"
              className="w-full"
            />
          )}
        </div>

        {/* Detected faces badges */}
        {faces.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {faces.map((face, i) => {
              const isKnown = face.name !== "unknown" && face.name !== "inconnu";
              return (
                <Badge key={i} variant={isKnown ? "default" : "destructive"}>
                  {face.name} — {Math.round(face.confidence * 100)}%
                </Badge>
              );
            })}
          </div>
        )}

        <canvas ref={captureCanvasRef} className="hidden" />
      </CardContent>
    </Card>
  );
}
