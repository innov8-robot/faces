import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { UserPlus, CheckCircle } from "lucide-react";
import WebcamCapture from "./WebcamCapture";
import { registerFace } from "@/lib/api";

interface Props {
  onRegistered: () => void;
}

export default function FaceRegister({ onRegistered }: Props) {
  const [name, setName] = useState("");
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const handleCapture = useCallback((blob: Blob) => {
    setImageBlob(blob);
    setPreview(URL.createObjectURL(blob));
    setMessage(null);
  }, []);

  const clearPreview = useCallback(() => {
    setImageBlob(null);
    setPreview(null);
  }, []);

  const handleRegister = async () => {
    if (!name.trim()) {
      setMessage({ type: "error", text: "Entrez un nom" });
      return;
    }
    if (!imageBlob) {
      setMessage({ type: "error", text: "Capturez ou uploadez une image" });
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const file = new File([imageBlob], "face.jpg", { type: "image/jpeg" });
      const result = await registerFace(file, name.trim());
      setMessage({ type: "success", text: result.message });
      setName("");
      setImageBlob(null);
      setPreview(null);
      onRegistered();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Erreur lors de l'enregistrement";
      if (typeof err === "object" && err !== null && "response" in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        setMessage({ type: "error", text: axiosErr.response?.data?.detail ?? msg });
      } else {
        setMessage({ type: "error", text: msg });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <UserPlus className="h-5 w-5" />
          Enregistrer un visage
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Nom</Label>
          <Input
            id="name"
            placeholder="Nom de la personne"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label>Photo</Label>
          <WebcamCapture
            onCapture={handleCapture}
            preview={preview}
            onClearPreview={clearPreview}
          />
        </div>

        <Button
          onClick={handleRegister}
          disabled={loading || !name.trim() || !imageBlob}
          className="w-full"
        >
          {loading ? "Enregistrement..." : "Enregistrer"}
        </Button>

        {message && (
          <Alert variant={message.type === "error" ? "destructive" : "default"}>
            {message.type === "success" && <CheckCircle className="h-4 w-4" />}
            <AlertDescription>{message.text}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
