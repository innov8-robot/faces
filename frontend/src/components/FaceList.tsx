import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Users, Trash2, RefreshCw } from "lucide-react";
import { listFaces, deleteFace, type RegisteredFace } from "@/lib/api";

interface Props {
  refreshTrigger: number;
}

export default function FaceList({ refreshTrigger }: Props) {
  const [faces, setFaces] = useState<RegisteredFace[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchFaces = async () => {
    setLoading(true);
    try {
      const data = await listFaces();
      setFaces(data);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFaces();
  }, [refreshTrigger]);

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Supprimer ${name} ?`)) return;
    try {
      await deleteFace(id);
      fetchFaces();
    } catch {
      // silently fail
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Visages enregistres
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={fetchFaces} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {faces.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">
            Aucun visage enregistre
          </p>
        ) : (
          <div className="space-y-1">
            {faces.map((face, i) => (
              <div key={face.id}>
                {i > 0 && <Separator className="my-1" />}
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm font-medium">{face.name}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleDelete(face.id, face.name)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
