import React, { useState } from 'react'
import "./sidebar.css"
import {FiUpload} from "react-icons/fi"
import {IoMdSettings} from "react-icons/io"
import {BsImages} from "react-icons/bs"

export default function Sidebar() {

    const [indexActiv, setIndexActiv] = useState(0)

    function handleUpload(){
        setIndexActiv(0)
    }

    function handleShow(){
        setIndexActiv(1)
    }

    function handleFilter(){
        setIndexActiv(2)
    }

    return (
    <div className="sidebar">
        <div className="sidebarWrapper">
            <div className="sidebarMenu"> {/* Only useful if we would add other menus */}
                <h3 className="sidebarTitle">Dashboard</h3>
                <hr className="horizontalLine"/>
                <ul className="sidebarList">
                    <li className= {indexActiv===0 ? "sidebarListItem active" : "sidebarListItem"} onClick={handleUpload}>
                        <FiUpload className="sidebarIcon"/>
                        Upload
                    </li>
                    <li className={indexActiv===1 ? "sidebarListItem active" : "sidebarListItem"}  onClick={handleShow}>
                        <BsImages className="sidebarIcon"/>
                        Similar Images
                    </li>
                    <li className={indexActiv===2 ? "sidebarListItem active" : "sidebarListItem"}  onClick={handleFilter}>
                        <IoMdSettings className="sidebarIcon"/>
                        Filter
                    </li>
                </ul>
            </div>
        </div>
    </div>
    )
}
