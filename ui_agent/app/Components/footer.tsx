'use client';
import { Dispatch, SetStateAction, useState } from "react";
import {  IBodyState } from "./body";

interface FooterProps {
    setMessage: Dispatch<SetStateAction<IBodyState>>
}

const Footer = (props: FooterProps) => {
    const [message, setMessage] = useState('');
    return (
        <footer className="w-80 bg-gray-500 p-2 rounded-lg border-blue-400 shadow-sm">
                <form className="flex flex-row justify-center items-center gap-2" onSubmit={(e) => {
                    e.preventDefault();
                    if (message.trim() === '') return;
                    const formData = new FormData();
                    formData.append('question', message);
                    fetch(
                        'http://localhost:8000/send_message/',
                        {
                            method: 'POST',
                            body: formData
                        }
                    )
                        .then((response) => response.json())
                        .then((data :string) => {
                            props.setMessage((prev) => ({
                                messages: [...prev.messages, {
                                    message: message,
                                    type: 'HUMAN'
                                }]
                            }));
                            props.setMessage((prev) => ({
                                messages: [...prev.messages, {
                                    message: data,
                                    type: 'AI'
                                }]
                            }));
                            setMessage('');
                        })
                        .catch((error) => {
                            console.error('Error:', error);
                        }
                    )
                }}>
                    <input type="text" value={message} onChange={(e) => {
                        setMessage(e.target.value);
                    }} placeholder="Ask me anything about your data..." className="w-[80%] border-gray-400 h-8 rounded-lg " />
                    <button className="px-4 py-2 bg-blue-500 text-white h-8 rounded-md"  >Send</button>
                </form>
        </footer>
    );
};
export default Footer;