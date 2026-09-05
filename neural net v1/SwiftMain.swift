//
//  SwiftMain.swift
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

import Foundation
import SwiftUI
import AppKit
import NeuralApp

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct SwiftMain: App  {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    var appEngine = AppEngine()

    var body: some Scene {
        WindowGroup {
            SwiftUIView(engine: appEngine)
            DrawingPad(engine: appEngine)
        }
    }
}
