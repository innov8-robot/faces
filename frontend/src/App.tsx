import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import FaceRegister from "@/components/FaceRegister";
import FaceList from "@/components/FaceList";
import LiveFeed from "@/components/LiveFeed";
import { ScanFace } from "lucide-react";

export default function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  return (
    <div className="dark min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-7xl p-6">
        <header className="mb-6 flex items-center gap-3">
          <ScanFace className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold">Face Recognition</h1>
        </header>

        <Tabs defaultValue="live" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="live">Live</TabsTrigger>
            <TabsTrigger value="manage">Gestion des visages</TabsTrigger>
          </TabsList>

          <TabsContent value="live">
            <LiveFeed />
          </TabsContent>

          <TabsContent value="manage">
            <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
              <FaceRegister
                onRegistered={() => setRefreshTrigger((n) => n + 1)}
              />
              <FaceList refreshTrigger={refreshTrigger} />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
