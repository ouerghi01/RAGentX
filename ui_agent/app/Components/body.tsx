'use client';

interface IBodyProps {
 message: string;
 type: string; // HUMAN OR AI 
}
export interface IBodyState { 
    messages:IBodyProps[];
}
import Image from "next/image";

const Body=(props:IBodyState)=>{
    return (
        <div className="w-80 h-96 bg-gray-500 p-2 rounded-lg shadow-sm">
            {
                props.messages && props.messages.map((msg:IBodyProps)=>{
                    return <div className="flex flex-row items-center gap-2">
                        <div className={`w-full flex ${msg.type === 'HUMAN' ? 'justify-end' : 'justify-start'}`}>
                            {
                                msg.type === 'HUMAN' ? 
                                <span className="text-white font-medium">Human</span> : 
                                <Image src="/logo_agent.jpg" alt="logo" width={50} height={50} className="rounded-lg" />
                                
                            }
                            <span className="text-white font-medium">{msg.message}</span>
                        </div>
                    </div>
                })
            }
         </div>
    )

}
export default Body;