import { Composition } from "remotion";
import { Lineup } from "./lineup";
import { PpoWin } from "./ppowin";
import { Matrix } from "./matrix";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="lineup"
        component={Lineup}
        durationInFrames={450}
        fps={30}
        width={1280}
        height={720}
      />
      <Composition
        id="ppo-win"
        component={PpoWin}
        durationInFrames={360}
        fps={30}
        width={1280}
        height={720}
      />
      <Composition
        id="matrix"
        component={Matrix}
        durationInFrames={450}
        fps={30}
        width={1280}
        height={720}
      />
    </>
  );
};
