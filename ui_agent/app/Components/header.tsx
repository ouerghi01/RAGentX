import { JSX } from "react";
import Image from "next/image";

const Header = (): JSX.Element => {
   
    return <>
    <header className="   w-80 bg-gray-500 p-2 rounded-lg shadow-sm">
        <div className="flex flex-row items-center gap-2">
            <Image src="/logo_agent.jpg" alt="logo" width={50} height={50} className="rounded-lg" />
            <span className="text-white font-medium">Agent</span>
        </div>
    </header>
   
   </>;
};
export default Header;