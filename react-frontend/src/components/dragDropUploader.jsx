import { FileUploader } from "react-drag-drop-files";

const fileTypes = ["JPG", "PNG", "GIF"];
const children = 
    <div className="py-4 px-6 my-2 space-x-2 flex items-center w-fit max-w-lg rounded-2xl shadow-xl text-center font-sans font-semibold text-lg bg-zinc-100 hover:bg-zinc-200 active:bg-zinc-300">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path d="M9.25 13.25a.75.75 0 001.5 0V4.636l2.955 3.129a.75.75 0 001.09-1.03l-4.25-4.5a.75.75 0 00-1.09 0l-4.25 4.5a.75.75 0 101.09 1.03L9.25 4.636v8.614z" />
            <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
        </svg>
        <h5 className="">
            UPLOAD
        </h5>
    </div>


export default function DragDropUploader(props) {
    return (
        // <FileUploader 
        //     handleChange={props.handleImageUploaded} 
        //     name="file" 
        //     types={fileTypes} 
        //     children={children}
        // />
        <input
            type="file" 
            accept="image/*"
            onChange={(e) => {console.log(e); props.handleImageUploaded(e.target.files[0])}}/>
    );
}